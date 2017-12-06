# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util, itertools
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  numGhosts = 2
  parts = []
  MIN_VALUE = -99999999
  MAX_VALUE = 99999999

  #Q-Learning stuff starts here ************************************************************************************
  weights = util.Counter()
  """
            {'len-eat-food': 4.080509109809522, 'num-walls-near-me': 0.40770090877317733,
            'dist-to-opp-1': 4.61012589104699, 'dist-to-opp-2': 6.935869511063713,
            'bias': 20.402545549047627, 'dist-to-partner': 0.6887876370970757,
            'num-capsules': 0.2842690373503767, 'score': 0.0, 'len-defend-food': 2.6039896922291317}
            """
  discount = 0.8
  epsilon = 0.05
  alpha = 0.02
  actionCount = 0

#
  def getQValue(self, gameState, action):
      qValue = 0
      features = self.getFeatures(gameState, action)
      for f in features:
          qValue += features[f] * self.weights[f]
      return qValue

  def update(self, gameState, action, nextState, reward):
      currentQValue = self.getQValue(gameState, action)
      actions = nextState.getLegalActions(self.index)
      features = self.getFeatures(gameState, action)
      maxQValue = self.computeValueFromQValues(gameState)
      difference = ( reward + (self.discount * maxQValue ) ) - currentQValue
      for f in features:
          self.weights[f] = self.weights[f] + (self.alpha * difference) * features[f]
      print "offensive weights:", self.weights
      print

  def getFeatures(self, gameState, action ):

      features = util.Counter()
      features["bias"] = 1.0

      defendFood = self.getFoodYouAreDefending(gameState).asList()
      eatFood = self.getFood(gameState).asList()
      capsules = gameState.getCapsules()
      walls = gameState.getWalls().asList()

      pacDistToOpp1=1.0
      pacDistToOpp2=1.0
      ghostDistToOpp1=1.0
      ghostDistToOpp2=1.0
      #exact feature numbers below
      if gameState.getAgentState(self.index).isPacman:
          pacDistToOpp1 = self.getMazeDistance(gameState.getAgentPosition(self.index), self.opponent1Position)
          pacDistToOpp2 = self.getMazeDistance(gameState.getAgentPosition(self.index), self.opponent2Position)
      else:
           ghostDistToOpp1 = self.getMazeDistance(gameState.getAgentPosition(self.index), self.opponent1Position)
           ghostDistToOpp2 = self.getMazeDistance(gameState.getAgentPosition(self.index), self.opponent2Position)

      distToPartner = self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(self.partnerIndex))
      lenDefendFood = len(defendFood)
      lenEatFood = len(eatFood)
      score = self.getScore(gameState)
      numWallsNearMe = 0
      numCapsules = len(capsules)

      for wall in walls:
          distance = util.manhattanDistance( gameState.getAgentPosition(self.index), wall )
          if distance == 1:
              numWallsNearMe += 1

      features["pac-dist-to-opp-1"] = 0.01 * ( (float)(pacDistToOpp1) )
      features["pac-dist-to-opp-2"] = 0.01 * ( (float)(pacDistToOpp2) )
      features["gh-dist-to-opp-1"] = 0.01 * ( (float)(ghostDistToOpp1) )
      features["gh-dist-to-opp-2"] = 0.01 * ( (float)(ghostDistToOpp2) )
      features["dist-to-partner"] = 0.01 * ( (float)(distToPartner) )
      features["len-defend-food"] = 0.01 * (float)(lenDefendFood)
      features["len-eat-food"] = 0.01 * (float)(lenEatFood)
      features["score"] = score
      features["num-walls-near-me"] = 0.01 * (float)(numWallsNearMe)
      features["num-capsules"] = 0.01 * (float)(numCapsules)

      return features

  def computeValueFromQValues(self, gameState):
      actions = gameState.getLegalActions(self.index)
      if len(actions) == 0:
          return 0.0
      bestValue = self.MIN_VALUE
      for action in actions:
          actionValue = self.getQValue(gameState, action)
          if actionValue > bestValue:
              bestValue = actionValue
      return bestValue

  def computeActionFromQValues(self, gameState):
      actions = gameState.getLegalActions(self.index)
      if len( actions ) == 0:
          return None
      bestValue = self.MIN_VALUE
      bestAction = 'Stop'
      for action in actions:
          actionValue = self.getQValue(gameState, action)
          if actionValue > bestValue:
              bestValue = actionValue
              bestAction = action
          elif actionValue == bestValue:
              bestAction = random.choice( [action, bestAction] )
      return bestAction

  def determineAction(self, gameState):
      actions = gameState.getLegalActions(self.index)

      if util.flipCoin(self.epsilon):
          return random.choice(actions)

      return self.computeActionFromQValues(gameState)

  def getReward(self, gameState):
      defendFood = self.getFoodYouAreDefending(gameState).asList()
      eatFood = self.getFood(gameState).asList()
      totalFood = len(defendFood) + len(eatFood)
      diffFood = len(eatFood) - len(defendFood)
      return 10*self.getScore(gameState) - len(eatFood) - 100*self.getMazeDistance(gameState.getAgentPosition(self.index),eatFood[len(eatFood)-1])




#Q-Learning stuff ends here ************************************************************************************


#Particle Filtering stuff starts here ************************************************************************************
  def getObservationDistribution(self, noisyDistance, gameState):
      observationDistributions = {}
      if noisyDistance == None:
          return util.Counter()
      if noisyDistance not in observationDistributions:
          distribution = util.Counter()
          for pos in self.legalPositions:
              dist = util.manhattanDistance(pos, gameState.getAgentPosition(self.index))
              prob = gameState.getDistanceProb(dist, noisyDistance)
              distribution[dist] += prob
              observationDistributions[noisyDistance] = distribution
      return observationDistributions[noisyDistance]

  def observeState(self, gameState):
      self.getBeliefDistribution()
      myPosition = gameState.getAgentPosition(self.index)
      noisyDistances = gameState.getAgentDistances()
      emissionModels = {}
      dist1 = noisyDistances[self.oppIndex1]
      emissionModels[self.oppIndex1] = self.getObservationDistribution(dist1, gameState)
      dist2 = noisyDistances[self.oppIndex2]
      emissionModels[self.oppIndex2] = self.getObservationDistribution(dist2, gameState)

      #GHOST PART IS LEFT OUT

      for belief in self.beliefs:
          prob = 1
          i = 0
          for ghostIndex in self.oppIndices:
              eM = emissionModels[ghostIndex]
              trueDistance = util.manhattanDistance(belief[i], myPosition)
              prob *= eM[trueDistance]
              self.beliefs[belief] = prob * self.beliefs[belief]
              i += 1

      if self.beliefs.totalCount() == 0:
          self.initializeParticles()
          self.getBeliefDistribution()

      finalList = []
      for k in range(self.numParticles):
          finalList.append(util.sample(self.beliefs))

      self.parts = finalList

  def getBeliefDistribution(self):
      "*** YOUR CODE HERE ***"
      #add 1 for each location pair, then normalize to get beliefs
      self.beliefs = util.Counter()
      for tuples in self.parts:
          self.beliefs[tuples] += 1

      self.beliefs.normalize()

  def initializeParticles(self):
      #because we need to shuffle the particles, use temporary list
      temporaryParticles = []
      #thank God for itertools
      for permutation in itertools.permutations(self.legalPositions, self.numGhosts):
          temporaryParticles.append(permutation)
      #shuffle changes temporaryParticles to a shuffled list of particles
      random.shuffle(temporaryParticles)
      self.numParticles = 700

      #moving the shuffled particles into our particle list
      i = 0
      j = 0
      self.parts = []
      while i < self.numParticles:
          self.parts.append(temporaryParticles[j])
          i+=1
          #need to go through the list multiple times to get the right number of particles
          j = (j+1)%len(temporaryParticles)

#Particle filtering stuff ends here ************************************************************************************

 #OffensiveAgent methods  starts here ************************************************************************************
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    #set up offensive agent & defensive agent indices
    self.indices = self.getTeam(gameState)
    self.oppIndices = self.getOpponents(gameState)
    self.index = self.indices[0]
    self.partnerIndex = self.indices[1]
    self.oppIndex1 = self.oppIndices[0]
    self.oppIndex2 = self.oppIndices[1]
    #set up legal positions
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

    #initialize particles
    self.initializeParticles()

  def chooseAction(self, gameState):
    self.actionCount += 1
    actions = gameState.getLegalActions(self.index)
    self.opponent1Position = gameState.getAgentPosition(self.oppIndex1)
    self.opponent2Position = gameState.getAgentPosition(self.oppIndex2)

    if self.opponent1Position == None or self.opponent2Position == None:
        self.observeState(gameState)
        self.getBeliefDistribution()

        maxProb = self.MIN_VALUE
        maxPositions = None
        for positions, prob in self.beliefs.items():
            if prob > maxProb:
                maxProb = prob
                maxPositions = positions
        self.opponent1Position = maxPositions[0]
        self.opponent2Position = maxPositions[1]

    maxPositionsList = [self.opponent1Position, self.opponent2Position]
    self.debugDraw(maxPositionsList, [0,0,1], clear = True)

    #now we get into q learning stuff
    bestAction = self.determineAction(gameState)
    nextGameState = gameState.generateSuccessor(self.index, bestAction)
    reward = self.getReward(gameState)
    self.update(gameState, bestAction, nextGameState, reward)
    return bestAction

 #OffensiveAgent methods end here ************************************************************************************

class DefensiveAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  numGhosts = 2
  parts = []
  MIN_VALUE = -99999999
  MAX_VALUE = 99999999

  #Q-Learning stuff starts here ************************************************************************************
  weights = {'bias': 16.570357793055432, 'dist-to-partner': 5.356309287248157, 'len-eat-food': 3.3140715586110434, 'num-walls-near-me': 0.6736262554850796, 'score': 4.580205030660764, 'len-defend-food': 90.93543185701286, 'dist-to-opp-1': 82.34732814235079, 'dist-to-opp-2': 20.750886129179438, 'num-capsules': 5.361597755188771}
  #defensive weights: {'bias': 21.95503818500934, 'dist-to-partner': 0.7075593066638088, 'len-eat-food': 4.391007637001868, 'num-walls-near-me': 0.43048629212645023, 'score': 0.0, 'len-defend-food': 1.6526545535210564, 'dist-to-opp-1': 1.529598121468612, 'dist-to-opp-2': 3.98751660752516, 'num-capsules': 0.2170382246306381}
  discount = 0.8
  epsilon = 0.05
  alpha = 0.02
  actionCount = 0

#
  def getQValue(self, gameState, action):
      qValue = 0
      features = self.getFeatures(gameState, action)
      for f in features:
          qValue += features[f] * self.weights[f]
      return qValue

  def update(self, gameState, action, nextState, reward):
      currentQValue = self.getQValue(gameState, action)
      actions = nextState.getLegalActions(self.index)
      features = self.getFeatures(gameState, action)
      maxQValue = self.computeValueFromQValues(gameState)
      difference = ( reward + (self.discount * maxQValue ) ) - currentQValue
      for f in features:
          self.weights[f] = self.weights[f] + (self.alpha * difference) * features[f]
      print "defensive weights: ", self.weights
      print

  def getFeatures(self, gameState, action ):

      features = util.Counter()
      features["bias"] = 1.0

      defendFood = self.getFoodYouAreDefending(gameState).asList()
      eatFood = self.getFood(gameState).asList()
      capsules = gameState.getCapsules()
      walls = gameState.getWalls().asList()

      #exact feature numbers below
      distToOpp1 = self.getMazeDistance(gameState.getAgentPosition(self.index), self.opponent1Position)
      distToOpp2 = self.getMazeDistance(gameState.getAgentPosition(self.index), self.opponent2Position)
      distToPartner = self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(self.partnerIndex))
      lenDefendFood = len(defendFood)
      lenEatFood = len(eatFood)
      score = self.getScore(gameState)
      numWallsNearMe = 0
      numCapsules = len(capsules)

      for wall in walls:
          distance = util.manhattanDistance( gameState.getAgentPosition(self.index), wall )
          if distance == 1:
              numWallsNearMe += 1

      features["dist-to-opp-1"] = 0.01 * ( (float)(distToOpp1) )
      features["dist-to-opp-2"] = 0.01 * ( (float)(distToOpp2) )
      features["dist-to-partner"] = 0.01 * ( (float)(distToPartner) )
      features["len-defend-food"] = 0.01 * (float)(lenDefendFood)
      features["len-eat-food"] = 0.01 * (float)(lenEatFood)
      features["score"] = score
      features["num-walls-near-me"] = 0.01 * (float)(numWallsNearMe)
      features["num-capsules"] = 0.01 * (float)(numCapsules)

      return features

  def computeValueFromQValues(self, gameState):
      actions = gameState.getLegalActions(self.index)
      if len(actions) == 0:
          return 0.0
      bestValue = self.MIN_VALUE
      for action in actions:
          actionValue = self.getQValue(gameState, action)
          if actionValue > bestValue:
              bestValue = actionValue
      return bestValue

  def computeActionFromQValues(self, gameState):
      actions = gameState.getLegalActions(self.index)
      if len( actions ) == 0:
          return None
      bestValue = self.MIN_VALUE
      bestAction = 'Stop'
      for action in actions:
          actionValue = self.getQValue(gameState, action)
          if actionValue > bestValue:
              bestValue = actionValue
              bestAction = action
          elif actionValue == bestValue:
              bestAction = random.choice( [action, bestAction] )
      return bestAction

  def determineAction(self, gameState):
      actions = gameState.getLegalActions(self.index)

      if util.flipCoin(self.epsilon):
          return random.choice(actions)

      return self.computeActionFromQValues(gameState)

  def getReward(self, gameState):
      defendFood = self.getFoodYouAreDefending(gameState).asList()
      eatFood = self.getFood(gameState).asList()
      totalFood = len(defendFood) + len(eatFood)
      diffFood = len(eatFood) - len(defendFood)
      return 10*self.getScore(gameState) + len(defendFood)




#Q-Learning stuff ends here ************************************************************************************


#Particle Filtering stuff starts here ************************************************************************************
  def getObservationDistribution(self, noisyDistance, gameState):
      observationDistributions = {}
      if noisyDistance == None:
          return util.Counter()
      if noisyDistance not in observationDistributions:
          distribution = util.Counter()
          for pos in self.legalPositions:
              dist = util.manhattanDistance(pos, gameState.getAgentPosition(self.index))
              prob = gameState.getDistanceProb(dist, noisyDistance)
              distribution[dist] += prob
              observationDistributions[noisyDistance] = distribution
      return observationDistributions[noisyDistance]

  def observeState(self, gameState):
      self.getBeliefDistribution()
      myPosition = gameState.getAgentPosition(self.index)
      noisyDistances = gameState.getAgentDistances()
      emissionModels = {}
      dist1 = noisyDistances[self.oppIndex1]
      emissionModels[self.oppIndex1] = self.getObservationDistribution(dist1, gameState)
      dist2 = noisyDistances[self.oppIndex2]
      emissionModels[self.oppIndex2] = self.getObservationDistribution(dist2, gameState)

      #GHOST PART IS LEFT OUT

      for belief in self.beliefs:
          prob = 1
          i = 0
          for ghostIndex in self.oppIndices:
              eM = emissionModels[ghostIndex]
              trueDistance = util.manhattanDistance(belief[i], myPosition)
              prob *= eM[trueDistance]
              self.beliefs[belief] = prob * self.beliefs[belief]
              i += 1

      if self.beliefs.totalCount() == 0:
          self.initializeParticles()
          self.getBeliefDistribution()

      finalList = []
      for k in range(self.numParticles):
          finalList.append(util.sample(self.beliefs))

      self.parts = finalList

  def getBeliefDistribution(self):
      "*** YOUR CODE HERE ***"
      #add 1 for each location pair, then normalize to get beliefs
      self.beliefs = util.Counter()
      for tuples in self.parts:
          self.beliefs[tuples] += 1

      self.beliefs.normalize()

  def initializeParticles(self):
      #because we need to shuffle the particles, use temporary list
      temporaryParticles = []
      #thank God for itertools
      for permutation in itertools.permutations(self.legalPositions, self.numGhosts):
          temporaryParticles.append(permutation)
      #shuffle changes temporaryParticles to a shuffled list of particles
      random.shuffle(temporaryParticles)
      self.numParticles = 700

      #moving the shuffled particles into our particle list
      i = 0
      j = 0
      self.parts = []
      while i < self.numParticles:
          self.parts.append(temporaryParticles[j])
          i+=1
          #need to go through the list multiple times to get the right number of particles
          j = (j+1)%len(temporaryParticles)

#Particle filtering stuff ends here ************************************************************************************

 #OffensiveAgent methods  starts here ************************************************************************************
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    #set up offensive agent & defensive agent indices
    self.indices = self.getTeam(gameState)
    self.oppIndices = self.getOpponents(gameState)
    self.index = self.indices[1]
    self.partnerIndex = self.indices[0]
    self.oppIndex1 = self.oppIndices[0]
    self.oppIndex2 = self.oppIndices[1]
    #set up legal positions
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

    #initialize particles
    self.initializeParticles()

  def chooseAction(self, gameState):


    self.actionCount += 1
    actions = gameState.getLegalActions(self.index)
    self.opponent1Position = gameState.getAgentPosition(self.oppIndex1)
    self.opponent2Position = gameState.getAgentPosition(self.oppIndex2)

    if self.opponent1Position == None or self.opponent2Position == None:
        self.observeState(gameState)
        self.getBeliefDistribution()

        maxProb = self.MIN_VALUE
        maxPositions = None
        for positions, prob in self.beliefs.items():
            if prob > maxProb:
                maxProb = prob
                maxPositions = positions
        self.opponent1Position = maxPositions[0]
        self.opponent2Position = maxPositions[1]

    maxPositionsList = [self.opponent1Position, self.opponent2Position]
    self.debugDraw(maxPositionsList, [0,0,1], clear = True)

    #now we get into q learning stuff
    bestAction = self.determineAction(gameState)
    nextGameState = gameState.generateSuccessor(self.index, bestAction)
    reward = self.getReward(gameState)
    self.update(gameState, bestAction, nextGameState, reward)
    return bestAction

 #OffensiveAgent methods end here ************************************************************************************
