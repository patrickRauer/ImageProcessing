from datetime import datetime


class StepStats:
    name = ""
    information = ''
    time = None

    def __init__(self, name, information):
        self.name = name
        self.information = information
        self.time = datetime.now()

    def __str__(self):
        if type(self.information) == tuple:
            out = '{}\t({}):\t{} ± {}'.format(self.name,
                                              self.time.strftime("%Y-%m-%d %H:%M:%S"),
                                              round(self.information[0], 2),
                                              round(self.information[1], 2))
        else:
            out = '{}\t({}):\t{}'.format(self.name,
                                         self.time.strftime("%Y-%m-%d %H:%M:%S"),
                                         self.information)
        return out


class AstrometryStats:
    steps = []

    def __init__(self):
        pass

    def add(self, name, information):
        s = StepStats(name, information)
        self.steps.append(s)

    def __str__(self):
        out = ''
        for s in self.steps:
            out += '{}\n'.format(str(s))
        return out
