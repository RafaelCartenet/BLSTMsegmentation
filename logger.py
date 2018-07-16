# Logging class

# Global libraries
import time
import datetime


class Logger:

    Logs= []

    def __init__(self, VERBOSE=0):
        self.VERBOSE= VERBOSE

    def write_log(self, log, time=None):
        if self.VERBOSE:
            print log
        else:
            pass

    def time(self, detail, duration):
        log= {
            'detail': detail,
            'duration': duration,
            'time': time.strftime("%H:%M:%S", time.gmtime(time.time()))
        }
        self.Logs.append(log)

    def write_all_logs(self):
        print "Time\t| Duration\t| Detail"
        for i,log in enumerate(self.Logs):
            s= "%s| %.4fs\t| %s"%\
                (log['time'], log['duration'], log['detail'])
            print s
