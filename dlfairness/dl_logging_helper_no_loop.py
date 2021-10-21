from timeit import default_timer as timer


class Logger:
    def __init__(self):
        logger_path = './' + 'DLVarLogNoLoop.csv'

        with open(logger_path, "r") as log_file:
            headers = log_file.readline().strip()

        headers = headers.split(',')
        self.headers = headers[2:]

        self.loggerOut = open(logger_path, "a")
        self.time = 0.0
        self.begin = 0.0
        self.iter = 0
        self.metric_dic = {}

    def log(self, key, value):

        if len(self.metric_dic) == 0:
            # first metric of the iteration
            self.begin = timer()
        elif key in self.metric_dic:
            # last metric of the iteration (duplication)
            self.write_metrics()

        self.metric_dic[key] = value

        if len(self.metric_dic) == len(self.headers):
            self.write_metrics()

    def write_metrics(self):
        end = timer()
        self.time = self.time + (end - self.begin)
        self.loggerOut.write("%s, %.5f" % (str(self.iter), self.time))
        self.time = 0.0
        self.iter = self.iter + 1
        for header in self.headers:
            if header in self.metric_dic:
                self.loggerOut.write(",%.5f" % self.metric_dic[header])
            else:
                self.loggerOut.write(",")
        self.loggerOut.write("\n")
        self.loggerOut.flush()
        self.metric_dic = {}

    def endLogger(self):
        self.loggerOut.close()


DLVarLogger = Logger()
