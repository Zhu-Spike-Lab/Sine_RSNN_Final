from classes.ATan1 import ATan1
#used by atan1 function in helper1.py
class ATanSurrogate1:
    def __init__(self, alpha=2.0):
        self.alpha = alpha

    def __call__(self, x):
        return ATan1.apply(x, self.alpha)