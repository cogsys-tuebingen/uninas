from uninas.utils.args import MetaArgument
from uninas.methods.abstract_method import AbstractMethod
from uninas.register import Register


@Register.method()
class RetrainMethod(AbstractMethod):
    """
    Standard training of a network
    """

    @classmethod
    def meta_args_to_add(cls, **__) -> [MetaArgument]:
        """
        list meta arguments to add to argparse for when this class is chosen,
        classes specified in meta arguments may have their own respective arguments
        """
        return super().meta_args_to_add(search=False, **__)
