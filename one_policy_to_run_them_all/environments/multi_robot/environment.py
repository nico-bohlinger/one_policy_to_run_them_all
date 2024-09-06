class TrainAndEvalEnvironment():
    def __init__(self, train_env, eval_env):
        self.train_env = train_env
        self.eval_env = eval_env

    def close(self):
        self.train_env.close()
        self.eval_env.close()


class DummyEnvironment():
    def close(self):
        pass
