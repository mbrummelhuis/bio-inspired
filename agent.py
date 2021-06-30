class Agent():
    def __init__(self,env):
        self.action_space = env.action_space
    
    def generateAction(self,env):
        action = env.action_space.sample()
        return action