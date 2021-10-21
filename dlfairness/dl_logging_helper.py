from timeit import default_timer as timer


class Logger:
    def __init__(self):
        logger_path = './' + 'DLVarLog.csv'

        with open(logger_path, "r") as log_file:
            headers = log_file.readline().strip()

        headers = headers.split(',')
        self.headers = headers[2:]

        self.loggerOut = open(logger_path, "a")
        self.time = 0.0
        self.begin = 0.0
        self.iter = 0.0
        self.metric_dic = {}

    def beginLoop(self, iter):
        self.begin = timer()
        self.iter = iter

    def endLoop(self):
        if self.metric_dic:
            end = timer()
            self.time = self.time + (end - self.begin)

            self.loggerOut.write("%s, %.5f" % (str(self.iter), self.time))
            self.time = 0.0
            self.iter = 0.0

            for header in self.headers:
                if header in self.metric_dic:
                    self.loggerOut.write(",%.5f" % self.metric_dic[header])
                else:
                    self.loggerOut.write(",")
            self.loggerOut.write("\n")
            #self.loggerOut.flush()

            self.metric_dic = {}

    def log(self, key, value):
        self.metric_dic[key] = value

    def endLogger(self):
        self.loggerOut.close()


DLVarLogger = Logger()
