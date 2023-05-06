import psutil


class Monitor(object):
    def __init__(self, path):
        self.path = path
        self.file_name = self.path + '/logs.txt'
        with open(self.file_name, "w") as f:
            f.write("Fuzzing Records: \n\n")

    def logging(self, node):
        if not isinstance(node, str):
            node = str(node)

        _, _, cpuInfo = self.getCPUState()
        _, _, _, MemoryInfo = self.getMemoryState()

        with open(self.file_name, "a") as f:
            f.write("node " + str(node) + ":")
            f.write("\n")
            f.write(cpuInfo)
            f.write("\n")
            f.write(MemoryInfo)
            f.write("\n")
            f.write("\n")

    def getCPUState(self, interval=1):
        """ function of Get CPU State """
        cpuCount = psutil.cpu_count()
        cpuPercent = psutil.cpu_percent(interval)
        logInfo = "Logic CPU: %s; CPU: %s" % (str(cpuCount), str(cpuPercent))

        return cpuCount, cpuPercent, logInfo

    def getMemoryState(self):
        """ function of GetMemory """
        phymem = psutil.virtual_memory()
        usedmem = int(phymem.used / 1024 / 1024)
        totalmem = int(phymem.total / 1024 / 1024)
        phymemPercent = "{:.2f}".format(float(usedmem / totalmem * 100))
        logInfo = "Memory used: %sM; Memory total: %sM; Memory percent: %s%%" % \
                  (str(usedmem), str(totalmem), str(phymemPercent))
        return usedmem, totalmem, phymemPercent, logInfo
