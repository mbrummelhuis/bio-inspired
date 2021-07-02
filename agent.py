class Agent():
    def __init__(self,env):
        self.name = 'Continuous agent'
    
    def getRandomAction(self, env):
        # Sample a random action from the action space
        action = env.action_space.sample()
        return action
    
    def getAction(self,state):
        # Sample an action based on the policy and state

        return action