"""
The base module for optimisers. This module contains the base class for all optimisers, which defines the interface for the optimisers. Each optimiser will inherit from this base class and implement the required methods. 
"""
import time
from abc import ABC, abstractmethod
import math


class Optimiser(ABC):
    def __init__(self, config):
        self.pop_size = config["pop_size"]
        self.max_iter = config.get("max_iter")
        self.dim      = config["dim"]
        self.bound    = config["bounds"]

        self.max_call = config.get("max_call")
        self.max_time = config.get("max_time")
        self.patience = config.get("patient")
        self.min_delta = config.get("min_delta",0)




    def _iter_loop(self):

        i = 0
        while self.max_iter is None or i < self.max_iter:
            yield i
            i = i+1


    def _max_iter(self):
        if self.max_iter is None:
            return math.inf
        else:
            self.max_iter


    def _should_stop(selfself, start_time, calls_sinces, best_history):

        def print_stop():
            print(" --------------------------------------------------- ")

        if self.max_time is not None and (time.time() - start_time ) >= self.max_time:
            print_stop()
            print(f"Stopping: max_time {self.max_time} seconds reached\n")
            return True

        if self.max_call is not None and calls_sinces_start > self.max_calls:
            print_stop()
            print(f"Stopping: max_calls {self.max_calls} reached\n")
            return True

        if self.patience is not None and len(best_history) > self.patience:
            window = best_history[-self.patience:]
            if max(window) - window[0] < self.min_delta:
                print_stop()
                print(f"Stopping: no improvement of <{self.min_delta} in last {self.patience} iterations\n")
                return True
        return False

    @abstractmethod
    def optimise(self, bot):
        pass

    def __str__(self):
        return self.__class__.__name__



