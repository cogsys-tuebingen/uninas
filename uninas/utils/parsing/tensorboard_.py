import glob
from uninas.register import Register


try:
    from tensorboard.backend.event_processing import event_accumulator


    class TbEvent:
        def __init__(self, value: float, step: int):
            self.value = value
            self.step = step

        def __str__(self):
            return str(self.value)


    def read_event_files(event_filenames: [str]) -> {str: [TbEvent]}:
        """
        read tensorboard files, returns all scalar events
        adapted from https://github.com/mrahtz/tbplot/blob/master/tbplot
        """
        events = {}
        for fn in event_filenames:
            try:
                ea = event_accumulator.EventAccumulator(fn)
                ea.Reload()
                for tag in ea.Tags()['scalars']:
                    events[tag] = []
                    for scalar in ea.Scalars(tag):
                        events[tag].append(TbEvent(scalar.value, scalar.step))
            except Exception as e:
                print(f"While reading '{fn}':", e)
        return events

    def find_tb_files(dir_: str) -> [str]:
        return glob.glob('%s/**/events.out.tfevents.*' % dir_, recursive=True)


except ImportError as ie:
    Register.missing_import(ie)
