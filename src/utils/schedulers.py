class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        arg1 = self.step_num ** (-0.5)
        arg2 = self.step_num * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

    def state_dict(self):
        return {
            'step_num': self.step_num,
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict['step_num']
        self.d_model = state_dict['d_model']
        self.warmup_steps = state_dict['warmup_steps']
