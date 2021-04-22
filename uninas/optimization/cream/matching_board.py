from typing import Callable, Union
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from uninas.models.networks.abstract import AbstractNetwork
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.optimization.estimators.abstract import AbstractEstimator
from uninas.training.optimizers.abstract import WrappedOptimizer
from uninas.utils.shape import Shape


class BoardEntry:
    def __init__(self, arc: [int], values: [float], input_data: torch.Tensor, sm_data: torch.Tensor):
        self.arc = arc
        self.values = None
        self.input_data = None
        self.sm_data = None
        self.update(values, input_data, sm_data)

    def update(self, values: [float], input_data: torch.Tensor, sm_data: torch.Tensor):
        self.values = values
        self.input_data = input_data
        self.sm_data = sm_data

    def __str__(self) -> str:
        return '%s([%s], %s)' % (self.__class__.__name__, ', '.join(['%.3f' % v for v in self.values]), str(self.arc))


class Board:
    _nds = NonDominatedSorting()

    def __init__(self, max_size: int, track_tensors: bool, is_for_best=True):
        self.max_size = max_size
        self.track_tensors = track_tensors
        self.is_for_best = is_for_best
        self.entries = []

    def get_entries(self) -> [BoardEntry]:
        return self.entries

    def get_entry_by_arc(self, arc: [int]) -> Union[BoardEntry, None]:
        """ if an entry has the given architecture, return it """
        for entry in self.entries:
            if entry.arc == arc:
                return entry
        return None

    def size(self) -> int:
        """ number of current entries """
        return len(self.entries)

    def num_values(self) -> int:
        """ num values that each entry has """
        return len(self.entries[0].values)

    def is_empty(self) -> bool:
        return self.size() == 0

    def is_full(self) -> bool:
        return self.size() >= self.max_size

    def is_overly_full(self) -> bool:
        return self.size() > self.max_size

    def remove_out_of_constraints(self, estimators: [AbstractEstimator]):
        """
        remove all board entries that are out of constraints,
        keep all those that violate the least constraints (preferably none)
        """
        num = [sum([e.get_constraint_badness(e.evaluate_tuple(entry.arc)) for e in estimators])
               for entry in self.entries]
        if (len(num) > 0) and (len(estimators) > 0) and (min(num) != max(num)):
            best = min(num)
            to_remove = [i for i, n in enumerate(num) if n > best]
            for n in sorted(to_remove, reverse=True):
                del self.entries[n]

    def _get_worst_entry(self) -> (int, int):
        """
        get the index+rank of worst dominated candidate
        (if the board tracks pareto best entries, otherwise give index+rank of a pareto-best candidate)
        """
        all_values = np.zeros(shape=(self.size(), self.num_values()))
        for i, e in enumerate(self.entries):
            all_values[i] = e.values
        _, rank = self._nds.do(all_values, return_rank=True)
        idx = int(np.argmax(rank) if self.is_for_best else np.argmin(rank))
        return idx, rank[idx]

    def update_board(self, arc: [int], values: [float], match_batch_size: int,
                     inputs: torch.Tensor, outputs: [torch.Tensor], teacher_outputs: [torch.Tensor] = None):
        """
        maybe add a path to the board

        :param arc: architecture of the current candidate
        :param values: metric values of the current candidate, smaller is always better
        :param match_batch_size: batch size of saved input+output
        :param inputs: inputs with which the metrics were measured
        :param outputs: outputs with which the metrics were measured
        :param teacher_outputs: outputs of the teacher of the current architecture
        """
        saved_inputs, saved_outputs = None, None
        if self.track_tensors:
            saved_inputs = inputs[:match_batch_size].detach().clone()
            o = outputs if (self.is_empty() or teacher_outputs is None) else teacher_outputs
            saved_outputs = F.softmax(o[-1][:match_batch_size].detach().clone(), 1)

        # if an entry with the given architecture already exists, just update its values
        entry = self.get_entry_by_arc(arc)
        if entry is None:
            entry = BoardEntry(arc, values, saved_inputs, saved_outputs)
            self.entries.append(entry)
        else:
            entry.update(values, saved_inputs, saved_outputs)
        self.entries = sorted(self.entries, key=lambda e: e.values[0])

        if self.is_overly_full():
            # try to remove a (by now) dominated solution
            # if none is found (all entries are pareto-best, rank == 0), remove the one with worst value1

            to_delete, rank = self._get_worst_entry()
            if (rank == 0) and self.is_for_best:
                self.entries = sorted(self.entries, key=lambda e: e.values[0], reverse=False)
                to_delete = -1

            del self.entries[to_delete]


class PrioritizedMatchingBoard(nn.Module):
    def __init__(self, board_size: int, grace_epochs: int, select_strategy: str, select_update_iter: int,
                 label_shape: Shape, match_batch_size: int, average_mmn_batches=False,
                 mmn_batch_size=-1, clip_grad_norm_value=1,
                 matching_net: Union[nn.Module, AbstractNetwork, None] = None):
        """
        a board of prioritized paths for direct NAS,
        also directly includes the meta weight updating

        based on:
            https://arxiv.org/pdf/2010.15821.pdf
            https://github.com/microsoft/cream

        :param board_size: number of entries (which are assumed pareto-optimal after some updates)
        :param grace_epochs: epochs passed before filling the board
        :param select_strategy: how to sample a teacher model
            value1: always pick the one with the highest value1
            random: always pick a random path
            l1: pick the most similar architecture by number of identical paths
            meta: pick the best match by output similarity
        :param select_update_iter: if the update strategy is 'meta', update the matching every n batches
        :param label_shape:
        :param match_batch_size: mini-batch size for student-teacher matching
        :param average_mmn_batches: whether to average the MMN results across the batches, or concat the inputs
        :param mmn_batch_size: mini-batch size for the training of the matching network
        :param clip_grad_norm_value: clip the probing step
        :param matching_net: matching network, use a simple nn.Linear if required but not given
        """
        super().__init__()
        self.grace_epochs = grace_epochs
        self.select_strategy = select_strategy
        self.select_update_iter = select_update_iter
        self.match_batch_size = match_batch_size
        self.average_mmn_batches = average_mmn_batches
        self.mmn_batch_size = mmn_batch_size
        self.clip_grad_norm_value = clip_grad_norm_value
        self.matching_net = None

        if select_strategy == "meta":
            assert match_batch_size * 2 <= mmn_batch_size,\
                "the MMN batch size (%d) must be at least twice as big as the match batch size (%d)"\
                % (mmn_batch_size, match_batch_size)

            size = label_shape.numel() if average_mmn_batches else label_shape.numel() * match_batch_size
            if matching_net is None:
                self.matching_net = nn.Linear(size, 1)
            elif isinstance(matching_net, AbstractNetwork):
                self.matching_net = matching_net
                self.matching_net.build(Shape([size]), Shape([1]))
            elif isinstance(matching_net, nn.Module):
                self.matching_net = matching_net
                out = self.matching_net(torch.zeros(size=[1, size]))
                assert out.shape == torch.Size([1, 1])
            else:
                raise NotImplementedError("can not handle matching net of type %s" % type(matching_net))

        self.pareto_best = Board(board_size, track_tensors=(select_strategy == 'meta'), is_for_best=True)
        self.pareto_worst = Board(board_size, track_tensors=(select_strategy == 'meta'), is_for_best=False)

    def forward(self, model_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        x = model_out - teacher_out
        y = self.matching_net(x if self.average_mmn_batches else x.view(1, -1))
        # networks may have multiple heads/outputs, only care for the final one
        if isinstance(y, list):
            y = y[-1]
        # possibly average over multiple
        if self.average_mmn_batches:
            y = y.mean(dim=0)
        return y.squeeze()

    def get_pareto_best(self) -> Board:
        return self.pareto_best

    def get_pareto_worst(self) -> Board:
        return self.pareto_worst

    def is_in_grace_time(self, epoch: int) -> bool:
        return epoch < self.grace_epochs

    def select_teacher(self, net: SearchUninasNetwork, arc: [int]) -> (float, [int]):
        """
        select a teacher architecture
        returns a matching_value in [0, 1] for the loss, and the teacher architecture
        """
        entries = self.pareto_best.get_entries()

        # the closest entry according to the meta matching network
        if self.select_strategy == 'meta':
            matching_value, teacher_arc = -1000000000, None
            for entry in entries:
                net.forward_strategy(fixed_arc=arc)
                out = net(entry.input_data)
                out = F.softmax(out[-1], dim=1)
                weight = self(out, entry.sm_data)
                if weight > matching_value:
                    matching_value = weight
                    teacher_arc = entry.arc
            return torch.sigmoid(matching_value), teacher_arc

        # any of the entries maximizing the number of same arc choices
        if self.select_strategy == 'l1':
            n_all = [sum([a0 == a1 for a0, a1 in zip(arc, entry.arc)]) for entry in entries]
            n_best = max(n_all)
            best = [entries[i] for i, n in enumerate(n_all) if n == n_best]
            return 0.5, random.choice(best).arc

        # the entry that maximizes value1
        if self.select_strategy == 'value1':
            return 0.5, entries[0].arc

        # a random entry
        if self.select_strategy == 'random':
            return 0.5, random.choice(entries).arc

        raise NotImplementedError

    def update_board(self, epoch: int, arc: [int], values: [float],
                     inputs: torch.Tensor, outputs: [torch.Tensor], teacher_outputs: [torch.Tensor] = None):
        """
        maybe add a path to the board

        :param epoch: current epoch
        :param arc: architecture of the current candidate
        :param values: metric values of the current candidate, smaller is always better
        :param inputs: inputs with which the metrics were measured
        :param outputs: outputs with which the metrics were measured
        :param teacher_outputs: outputs of the teacher of the current architecture
        """
        self.pareto_best.update_board(arc, values, match_batch_size=self.match_batch_size,
                                      inputs=inputs, outputs=outputs, teacher_outputs=teacher_outputs)
        self.pareto_worst.update_board(arc, values, match_batch_size=self.match_batch_size,
                                       inputs=inputs, outputs=outputs, teacher_outputs=teacher_outputs)

    def remove_out_of_constraints(self, estimators: [AbstractEstimator]):
        """
        remove all board entries that are out of constraints,
        keep all those that violate the least constraints (preferably none)
        """
        self.pareto_best.remove_out_of_constraints(estimators)
        self.pareto_worst.remove_out_of_constraints(estimators)

    @classmethod
    def _get_student_loss(cls, inputs: torch.Tensor, net: SearchUninasNetwork,
                          arc: [int], teacher_arc: [int], meta_value: float, loss_fn: Callable) -> torch.Tensor:
        net.forward_strategy(fixed_arc=arc)
        logits = net(inputs)
        with torch.no_grad():
            net.forward_strategy(fixed_arc=teacher_arc)
            teacher_logits = net(inputs)
            soft_target = F.softmax(teacher_logits[-1], dim=1)
        return meta_value * loss_fn(logits, soft_target)

    def _get_valid_loss(self, inputs: torch.Tensor, targets: torch.Tensor, net: SearchUninasNetwork,
                        arc: [int], loss_fn: Callable) -> torch.Tensor:
        x = inputs[self.match_batch_size:self.match_batch_size * 2].clone()
        y = targets[self.match_batch_size:self.match_batch_size * 2]
        assert x.shape[0] > 1, "too small MMN batch size for slice, %s, %s, %s" % (inputs.shape, x.shape, y.shape)
        net.forward_strategy(fixed_arc=arc)
        return loss_fn(net(x), y)

    @classmethod
    def _get_mmn_grads(cls, valid_loss: torch.Tensor, params_net: [nn.Parameter], params_mmn: [nn.Parameter],
                       one_student_weight: torch.Tensor) -> [torch.Tensor]:
        """ compute the 2nd order loss for the meta matching network's parameters """
        grads1_student = torch.autograd.grad(valid_loss, params_net, retain_graph=True, allow_unused=True)
        return torch.autograd.grad(one_student_weight, params_mmn, grad_outputs=grads1_student)

    def update_matching(self, inputs: torch.Tensor, target: torch.Tensor, arc: [int], net: SearchUninasNetwork,
                        opt_net: WrappedOptimizer, opt_mmn: WrappedOptimizer, loss_fn: Callable,
                        epoch: int, batch_index: int, current_lr_ratio=0.0, preserve_grads=False):
        """
        maybe update the matching weights

        :param inputs: network inputs
        :param target: target network outputs
        :param arc: current candidate architecture
        :param net: search network
        :param opt_net: optimizer for the search network weights
        :param opt_mmn: optimizer for the meta matching network weights
        :param loss_fn: criterion(outputs, targets)
        :param epoch:
        :param batch_index:
        :param current_lr_ratio: ratio of (current / init) learning rate, to keep opt_net/opt_mmn the same
        :param preserve_grads: whether to save and restore current gradients of the network parameters
        """
        if self.is_in_grace_time(epoch)\
                or self.pareto_best.is_empty()\
                or (((batch_index + 1) % self.select_update_iter) > 0)\
                or (self.select_strategy != 'meta'):
            return

        # save current gradients, weights
        saved_net_grads = opt_net.get_all_gradients(make_copy=True) if preserve_grads else None
        params_net = opt_net.get_all_weights()
        params_mmn = opt_mmn.get_all_weights()

        # possibly use only a slice of the inputs/targets, reducing the memory requirements
        if self.mmn_batch_size > 0:
            inputs = inputs[:self.mmn_batch_size]
            target = target[:self.mmn_batch_size]

        # follow the schedule of the real optimizer
        if isinstance(current_lr_ratio, float):
            initial_net = opt_net.param_groups[0]['initial_lr']
            opt_net.set_optimizer_lr(current_lr_ratio * initial_net, update_initial=False)
            initial_mmn = opt_mmn.param_groups[0]['initial_lr']
            opt_net.set_optimizer_lr(current_lr_ratio * initial_mmn, update_initial=False)

        # select teacher, use it to compute student loss and first order gradient
        meta_value, teacher_arc = self.select_teacher(net, arc)
        kd_loss = self._get_student_loss(inputs, net, arc, teacher_arc, meta_value, loss_fn)
        opt_net.zero_grad()
        grads1 = torch.autograd.grad(kd_loss, params_net, create_graph=True, allow_unused=True)
        del kd_loss, meta_value

        # simulate sgd update for one parameter, set all student gradients
        lr = opt_net.param_groups[-1]['lr']
        one_student_weight = None
        for p, g in zip(params_net, grads1):
            if (one_student_weight is None) and (g is not None):
                one_student_weight = p + g * lr
                assert p.shape == g.shape, "Mismatch of param (%s) and gradient (%s) shapes" % (p.shape, g.shape)
            p.grad = g
        del grads1

        # update student, get the validation loss
        torch.nn.utils.clip_grad_norm_(params_net, self.clip_grad_norm_value)
        opt_net.step()
        opt_net.zero_grad()
        valid_loss = self._get_valid_loss(inputs, target, net, arc, loss_fn)

        # compute second order gradient, update the meta matching network
        opt_net.zero_grad()
        opt_mmn.zero_grad()
        grads2 = self._get_mmn_grads(valid_loss, params_net, params_mmn, one_student_weight)
        for p, g in zip(params_mmn, grads2):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(params_mmn, self.clip_grad_norm_value)
        opt_mmn.step()
        del one_student_weight, valid_loss, grads2

        # restore saved gradients, remove mmn gradients
        if preserve_grads:
            for p, g in zip(params_net, saved_net_grads):
                p.grad = g
        for p in params_mmn:
            del p.grad
