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
  MIN_VALUE = -999999999999999999999
  MAX_VALUE = 9999999999999999999999

  #Q-Learning stuff starts here ************************************************************************************
  weights = {'food-near-me': 10.00, 'pac-dist-to-ghost-2': -804.31838007338729, 'pac-dist-to-ghost-1': 799.70264040440566, 'num-walls-near-me': -1219.7076297060364, 'bias': -92237.670776001629, 'num-capsules': -2053.5912941889956, 'pac-dist-to-opp-2': -7818.4887212910671, 'pac-dist-to-opp-1': -121.2907421163786, 'len-eat-food': -24949.774895975774, 'score': 3798.1831391978394, 'dist-to-partner': -121.8604403468308, 'len-defend-food': -6619.0517184995451}
  epsilon = 0.01
  alpha = 0.02
  discount = 0.8
  actionCount = 0

  gameCount = 0

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

  def getFeatures(self, gameState, action ):

      features = util.Counter()
      features["bias"] = 1.0

      nextS = gameState.generateSuccessor(self.index, action)
      defendFood = self.getFoodYouAreDefending(nextS).asList()
      eatFood = self.getFood(nextS).asList()
      capsules = nextS.getCapsules()
      walls = nextS.getWalls().asList()

      pacDistToOpp1=1.0
      pacDistToOpp2=1.0
      pacDistToGhost1 = 0.0
      pacDistToGhost2 = 0.0
      opp1IsPacman = gameState.getAgentState(self.oppIndex1).isPacman
      opp2IsPacman = gameState.getAgentState(self.oppIndex2).isPacman
      if opp1IsPacman == False:
          pacDistToGhost1 = self.getMazeDistance(gameState.getAgentPosition(self.index), self.opponent1Position)
      if opp2IsPacman == False:
          pacDistToGhost2 = self.getMazeDistance(gameState.getAgentPosition(self.index), self.opponent2Position)
      #exact feature numbers below

      pacDistToOpp1 = self.getMazeDistance(nextS.getAgentPosition(self.index), self.opponent1Position)
      pacDistToOpp2 = self.getMazeDistance(nextS.getAgentPosition(self.index), self.opponent2Position)
      #distToPartner = self.getMazeDistance(nextS.getAgentPosition(self.index), nextS.getAgentPosition(self.partnerIndex))
      lenDefendFood = len(defendFood)
      lenEatFood = len(eatFood)
      score = self.getScore(nextS)
      numWallsNearMe = 0
      foodNearMe = 0
      numCapsules = len(capsules)

      for wall in walls:
          distance = util.manhattanDistance(nextS.getAgentPosition(self.index), wall )
          if distance == 1:
              numWallsNearMe += 1
      for food in self.getFood(gameState):
          distance = util.manhattanDistance(nextS.getAgentPosition(self.index), food)
          if distance < 3:
              foodNearMe += 1


      features["pac-dist-to-opp-1"] = 0.01 * ( (float)(pacDistToOpp1) )
      features["pac-dist-to-opp-2"] = 0.01 * ( (float)(pacDistToOpp2) )
      features["pac-dist-to-ghost-1"] = 0.01 * ( (float)(pacDistToGhost1) )
      features["pac-dist-to-ghost-2"] = 0.01 * ( (float)(pacDistToGhost2) )
      #features["dist-to-partner"] = 0.01 * ( (float)(distToPartner) )
      features["len-defend-food"] = 0.01 * (float)(lenDefendFood)
      features["len-eat-food"] = 0.01 * (float)(lenEatFood)
      features["score"] = score
      features["num-walls-near-me"] = 0.01 * (float)(numWallsNearMe)
      features["num-capsules"] = 0.01 * (float)(numCapsules)
      features["food-near-me"] = (float)(foodNearMe)
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

      return 'Stop'

  def getReward(self, gameState):
      defendFood = self.getFoodYouAreDefending(gameState).asList()
      eatFood = self.getFood(gameState).asList()
      totalFood = len(defendFood) + len(eatFood)
      diffFood = len(eatFood) - len(defendFood)
      downgrade = 0
      if gameState.getAgentPosition(self.index) == self.initialposition:
          return -1000000
      if gameState.getAgentState(self.index).isPacman == False:
          downgrade = -100000000
      return (1000000*self.getScore(gameState)
             - 100000000*len(eatFood)
             - 10*self.getMazeDistance(gameState.getAgentPosition(self.index),eatFood[len(eatFood)-1])) + downgrade + len(defendFood)
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
    self.gameCount += 1
    print "*"*60
    print "game Count", self.gameCount
    print "*"*60
    print
    #set up offensive agent & defensive agent indices
    self.indices = self.getTeam(gameState)
    self.oppIndices = self.getOpponents(gameState)
    self.index = self.indices[0]
    self.partnerIndex = self.indices[1]
    self.oppIndex1 = self.oppIndices[0]
    self.oppIndex2 = self.oppIndices[1]
    self.initialposition = gameState.getAgentPosition(self.index)
    self.opp1InitialPosition = gameState.getAgentPosition(self.oppIndex1)
    self.opp2InitialPosition = gameState.getAgentPosition(self.oppIndex2)
    #set up legal positions
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

    #initialize particles
    self.initializeParticles()

  def chooseAction(self, gameState):

    self.actionCount += 1
    actions = gameState.getLegalActions(self.index)
    """
    opp1TruePos = gameState.getAgentPosition(self.oppIndex1)
    opp2TruePos = gameState.getAgentPosition(self.oppIndex2)

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

    if opp1TruePos != None:
        self.opponent1Position = opp1TruePos
    if opp2TruePos != None:
        self.opponent2Position = opp2TruePos

    maxPositionsList = [self.opponent1Position, self.opponent2Position]
    self.debugDraw(maxPositionsList, [0,0,1], clear = True)

    #now we get into q learning stuff
    bestAction = self.determineAction(gameState)
    nextGameState = gameState.generateSuccessor(self.index, bestAction)
    reward = self.getReward(gameState)
    self.update(gameState, bestAction, nextGameState, reward)
    """
    return random.choice(actions)

 #OffensiveAgent methods end here ************************************************************************************

class DefensiveAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  numGhosts = 2
  parts = []
  MIN_VALUE = -999999999999999999999
  MAX_VALUE = 999999999999999999999

  #Q-Learning stuff starts here ************************************************************************************
  weights = {'num-pellets-being-carried': -0.0, 'dist-to-best-spot': 9.998000399920016e-05, 'bias': 9.998000399920016e-05, 'closest-pacman-distance': 0.9998000399920016} 
  alpha = 0.02
  discount = 0.8
  actionCount = 0
  epsilon = 0.01

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
      difference = reward + (self.discount * maxQValue ) - currentQValue
      for f in features:
          self.weights[f] = self.weights[f] + (self.alpha * difference) * features[f]

      #normalizing the weights
      factor=1.0/sum(self.weights.itervalues())
      for feature in self.weights:
          self.weights[feature] = self.weights[feature]*factor

  def getFeatures(self, gameState, action ):
      features = util.Counter()
      features["bias"] = 1.0
      nextState = gameState.generateSuccessor(self.index, action)
      myPosition = nextState.getAgentPosition(self.index)

      #closest Pacman
      closestPacmanDistance = 0.0
      opp1IsPacman = False
      if nextState.getAgentState(self.oppIndex1).isPacman:
          opp1IsPacman = True
          distanceToPac = self.getMazeDistance( myPosition, self.opponent1Position )
          closestPacmanDistance = distanceToPac

      if nextState.getAgentState(self.oppIndex2).isPacman:
          distanceToPac = self.getMazeDistance( myPosition, self.opponent2Position )
          if opp1IsPacman:
              if distanceToPac < closestPacmanDistance:
                  closestPacmanDistance = distanceToPac
          else:
              closestPacmanDistance = distanceToPac

      #number of pellets being carried:
      numPellets = nextState.getAgentState(self.oppIndex1).numCarrying + nextState.getAgentState(self.oppIndex2).numCarrying

      #distance to best spot
      distanceToBestSpot = self.getMazeDistance(myPosition, self.bestSpot)

      if closestPacmanDistance != 0.0:
          features["closest-pacman-distance"] =  10000 / (float)(closestPacmanDistance)
      else:
          features["closest-pacman-distance"] =  10000

      features["num-pellets-being-carried"] = (float)(numPellets)

      if distanceToBestSpot == 0.0:
          features["dist-to-best-spot"] = 0.0
      else:
          features["dist-to-best-spot"] = ( 1.0 / (float)( distanceToBestSpot ) )

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
      bestAction = None
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
      opp1IsPacman = gameState.getAgentState(self.oppIndex1).isPacman
      opp2IsPacman = gameState.getAgentState(self.oppIndex2).isPacman

      for action in actions:
          nextState = gameState.generateSuccessor(self.index, action)
          nextPos = nextState.getAgentPosition(self.index)

          if opp1IsPacman and nextPos == self.opponent1Position:
              return action
          if opp2IsPacman and nextPos == self.opponent2Position:
              return action


      if util.flipCoin(self.epsilon):
          return random.choice(actions)

      return self.computeActionFromQValues(gameState)

  def getReward(self, gameState):

      myPosition = gameState.getAgentPosition(self.index)

      #closest Pacman
      closestPacmanDistance = 0.0
      opp1IsPacman = False
      opp2IsPacman = False
      noPacman = True
      if gameState.getAgentState(self.oppIndex1).isPacman:
          opp1IsPacman = True
          distanceToPac = self.getMazeDistance( myPosition, self.opponent1Position )
          closestPacmanDistance = distanceToPac
          noPacman = False

      if gameState.getAgentState(self.oppIndex2).isPacman:
          opp2IsPacman = True
          noPacman = False
          distanceToPac = self.getMazeDistance( myPosition, self.opponent2Position )
          if opp1IsPacman:
              if distanceToPac < closestPacmanDistance:
                  closestPacmanDistance = distanceToPac
          else:
              closestPacmanDistance = distanceToPac

      #maze distance to best spot
      distanceToBest = (float)( self.getMazeDistance(myPosition, self.bestSpot) )

      #give it rewards
        #if neither is pacman go to good spot
      if noPacman:
          if distanceToBest != 0.0:
              reward =  ( 1.0 / (float)( distanceToBest ) )
              return reward
          return 0.0
      else:
          if closestPacmanDistance != 0.0:
              return 10000 / float(closestPacmanDistance)
          return 10000

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
    print "defensive weights: ", self.weights
    #set up offensive agent & defensive agent indices
    self.indices = self.getTeam(gameState)
    self.oppIndices = self.getOpponents(gameState)
    self.index = self.indices[1]
    self.partnerIndex = self.indices[0]
    self.oppIndex1 = self.oppIndices[0]
    self.oppIndex2 = self.oppIndices[1]
    #set up legal positions
    self.initialposition = gameState.getAgentPosition(self.index)
    self.opp1InitialPosition = gameState.getAgentPosition(self.oppIndex1)
    self.opp2InitialPosition = gameState.getAgentPosition(self.oppIndex2)
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

    #set up bestSpot
    self.bestSpot = (12, 8)

    #initialize particles
    self.initializeParticles()

  def chooseAction(self, gameState):

    self.actionCount += 1
    """
    if self.actionCount <= 13:
        return 'North'
    if self.actionCount == 14 or self.actionCount == 15:
        return 'East'
    if self.actionCount <= 19:
        return 'South'
    """
    actions = gameState.getLegalActions(self.index)
    opp1TruePos = gameState.getAgentPosition(self.oppIndex1)
    opp2TruePos = gameState.getAgentPosition(self.oppIndex2)


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

    if opp1TruePos != None:
        self.opponent1Position = opp1TruePos
    if opp2TruePos != None:
        self.opponent2Position = opp2TruePos

    maxPositionsList = [self.opponent1Position, self.opponent2Position]
    self.debugDraw(maxPositionsList, [0,0,1], clear = True)

    #now we get into q learning stuff
    bestAction = self.determineAction(gameState)
    nextGameState = gameState.generateSuccessor(self.index, bestAction)
    reward = self.getReward(nextGameState)
    self.update(gameState, bestAction, nextGameState, reward)

    return bestAction

 #OffensiveAgent methods end here ************************************************************************************
