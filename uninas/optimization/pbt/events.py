
class AbstractPbtEvent:
    def __init__(self, epoch: int, client_id: int):
        self.epoch = epoch
        self.client_id = client_id

    def str_columns(self) -> [str]:
        raise NotImplementedError


class ReplacementPbtEvent(AbstractPbtEvent):
    def __init__(self, epoch: int, client_id: int,
                 epoch_replaced_with: int, client_replaced_with: int, path_replaced_with: str):
        super().__init__(epoch, client_id)
        self.epoch_replaced_with = epoch_replaced_with
        self.client_replaced_with = client_replaced_with
        self.path_replaced_with = path_replaced_with

    def str_columns(self) -> [str]:
        return ["Replacing",
                "(epoch=%d, client_id=%d)" % (self.epoch, self.client_id),
                "with",
                "(epoch=%d, client_id=%d)" % (self.epoch_replaced_with, self.client_replaced_with),
                "path=%s" % self.path_replaced_with]
