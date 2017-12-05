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
      self.numParticles = 1000

      #moving the shuffled particles into our particle list
      i = 0
      j = 0
      self.parts = []
      while i < self.numParticles:
          self.parts.append(temporaryParticles[j])
          i+=1
          #need to go through the list multiple times to get the right number of particles
          j = (j+1)%len(temporaryParticles)

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

    actions = gameState.getLegalActions(self.index)

    self.observeState(gameState)
    self.getBeliefDistribution()

    positionsSeen = util.Counter()
    for positions, prob in self.beliefs.items():
        positionsList = [positions[0], positions[1]]

        self.debugDraw(positionsList, [prob, 0, 0], clear = False)
    return random.choice(actions)

class DefensiveAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

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

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)
