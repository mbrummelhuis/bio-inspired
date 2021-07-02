class Agent():
    def __init__(self,env):
        self.name = 'Continuous agent'
        self.action_space = env.action_space

    def getRandomAction(self):
        # Sample a random action from the action space
        action = self.action_space.sample()
        return action
    
    def getAction(self,state,policy):
        # Sample an action based on the policy and state
        
        return action