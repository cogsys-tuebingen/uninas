import unittest
from uninas.optimization.cream.matching_board import PrioritizedMatchingBoard
from uninas.utils.shape import Shape


class TestCream(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_output_shapes(self):
        """
        expected output shapes of standard layers
        """
        batch_size = 32
        data_shape = Shape([3, 224, 224])
        label_shape = Shape([1000])
        data, label = data_shape.random_tensor(batch_size=batch_size), label_shape.random_tensor(batch_size=batch_size)

        prio = PrioritizedMatchingBoard(board_size=5, grace_epochs=0,
                                        select_strategy='l1', select_update_iter=0,
                                        label_shape=label_shape,
                                        match_batch_size=1, average_mmn_batches=False,
                                        mmn_batch_size=-1, clip_grad_norm_value=1)
        assert prio.pareto_best.is_empty()

        # basic inserting, optimal
        prio.update_board(epoch=0, arc=[0, 0], values=[1, 0, 0, 0, 0], inputs=data, outputs=[label])
        prio.update_board(epoch=0, arc=[0, 1], values=[0, 1, 0, 0, 0], inputs=data, outputs=[label])
        prio.update_board(epoch=0, arc=[0, 2], values=[0, 0, 0, 0, 0], inputs=data, outputs=[label])
        prio.update_board(epoch=0, arc=[0, 3], values=[0, 0, 0, 2, 0], inputs=data, outputs=[label])
        assert prio.pareto_best.size() == prio.pareto_worst.size() == 4
        assert not prio.pareto_best.is_empty()
        assert not prio.pareto_best.is_full()
        assert not prio.pareto_best.is_overly_full()
        assert prio.pareto_best.get_entry_by_arc([0, 0]) is not None

        # updating existing ones (for better or worse)
        prio.update_board(epoch=0, arc=[0, 2], values=[0, 0, 1, 0, 0], inputs=data, outputs=[label])
        prio.update_board(epoch=0, arc=[0, 3], values=[0, 0, 0, 1, 0], inputs=data, outputs=[label])
        assert prio.pareto_best.size() == prio.pareto_worst.size() == 4
        for entry in prio.pareto_best.get_entries():
            assert sum(entry.values) == 1

        # inserting a worse one, the board will be full then
        worst_arc = [1, 0]
        prio.update_board(epoch=0, arc=worst_arc, values=[1, 0, 0, 1, 0], inputs=data, outputs=[label])
        assert prio.pareto_best.size() == 5
        assert prio.pareto_best.is_full()
        assert not prio.pareto_best.is_overly_full()

        # inserting even worse ones, but the pareto-best board is already full with better solutions
        for arc, values in [
            ([1, 1], [1, 1, 0, 1, 0]),
            ([1, 2], [2, 0, 0, 1, 0]),
            ([1, 3], [1, 0, 0, 2, 0]),
            ([1, 4], [1, 0, 2, 1, 0]),
        ]:
            prio.update_board(epoch=0, arc=arc, values=values, inputs=data, outputs=[label])
            assert prio.pareto_best.size() == prio.pareto_worst.size() == 5
            assert prio.pareto_best.is_full()
            assert not prio.pareto_best.is_overly_full()
            assert prio.pareto_best.get_entry_by_arc(arc) is None
            assert prio.pareto_worst.get_entry_by_arc(arc) is not None

        # after adding these bad ones, the pareto worst front should not contain the good ones anymore
        for arc in ([0, 0], [0, 1], [0, 2], [0, 3]):
            assert prio.pareto_worst.get_entry_by_arc(arc) is None, "arc %s still there" % arc

        # worst arc is not part of the pareto front, but is still in there
        worst_entry_idx, worst_rank = prio.pareto_best._get_worst_entry()
        assert prio.pareto_best.get_entries()[worst_entry_idx].arc == worst_arc
        assert worst_rank == 1

        # insert another pareto optimal one, now only optimal solutions should exist
        prio.update_board(epoch=0, arc=[0, 4], values=[0, 0, 0, 0, 1], inputs=data, outputs=[label])
        assert prio.pareto_best.get_entry_by_arc(worst_arc) is None
        worst_entry_idx, worst_rank = prio.pareto_best._get_worst_entry()
        assert worst_rank == worst_entry_idx == 0


if __name__ == '__main__':
    unittest.main()
