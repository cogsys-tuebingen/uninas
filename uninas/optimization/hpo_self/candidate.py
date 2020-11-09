
class Candidate:
    _id_counter = 0
    _worst_rank = 10  # limit recursion depth

    def __init__(self, values: tuple, iteration: int = 0):
        self.id = self._id_counter
        Candidate._id_counter += 1
        self.values = tuple(values)
        self.iteration = iteration
        self.metrics = {}
        self.evaluated = False
        self.cur_ranked = -1
        self.cur_dominating = []
        self.cur_crowding_dist = 0
        self.cur_cdh = 0
        self.name = '{}[id={:<6} iteration={:<3} gene={}]'.format(self.__class__.__name__, self.id,
                                                                  self.iteration, str(self.values))

    def __str__(self):
        return '{name} rank={rank:<2} metrics=[{metrics}]'.format(**{
            'name': self.name,
            'rank': self.cur_ranked,
            'metrics': ', '.join(['%s=%s' % (k, v) for k, v in self.metrics.items()]),
        })

    def reset(self):
        self.cur_ranked = -1
        self.cur_dominating = []
        self.cur_crowding_dist = 0
        self.cur_cdh = 0

    def apply_rank(self, rank=0):
        if self._worst_rank >= rank > self.cur_ranked:
            self.cur_ranked = rank
            for candidate in self.cur_dominating:
                candidate.apply_rank(rank+1)
