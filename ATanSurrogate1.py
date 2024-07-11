from classes.ATan1 import ATan1
class ATanSurrogate1:
    def __init__(self, alpha=2.0):
        self.alpha = alpha

    def __call__(self, x):
        return ATan1.apply(x, self.alpha)