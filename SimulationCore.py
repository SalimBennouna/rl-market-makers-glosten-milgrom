import numpy as np
import pandas as pd

import os, queue, sys
from Message import MessageType

class SimulationCore:

  def __init__(self, kernel_name, random_state = None):
    label = kernel_name
    rng_seed = random_state
    self.name = label
    self.random_state = rng_seed
    if not rng_seed:
      raise ValueError("A valid, seeded np.random.RandomState object is required " + "for the Kernel", self.name)
      sys.exit()
    self.messages = queue.PriorityQueue()
    self.currentTime = None
    self.kernelWallClockStart = pd.Timestamp('now')
    self.meanResultByAgentType = {}
    self.agentCountByType = {}
    self.summaryLog = []

  def launch(self, agents = [], startTime = None, stopTime = None,
             num_simulations = 1, defaultComputationDelay = 1,
             skip_log = False, seed = None, oracle = None, log_dir = None):

    cohort = agents
    t_start = startTime
    t_stop = stopTime
    sim_count = num_simulations
    skip_flag = skip_log
    log_root = log_dir
    self.agents = cohort
    self.custom_state = {}
    self.startTime = t_start
    self.stopTime = t_stop
    self.seed = seed
    self.skip_log = skip_flag
    self.oracle = oracle
    if log_root:
      self.log_dir = log_root
    else:
      self.log_dir = str(int(self.kernelWallClockStart.timestamp()))
    self.agentCurrentTimes = [self.startTime] * len(cohort)

    for sim_idx in range(sim_count):
      for agent in self.agents:
        agent.kernelInitializing(self)
      for agent in self.agents:
        agent.kernelStarting(self.startTime)
      self.currentTime = self.startTime
      wall_start = pd.Timestamp('now')
      msg_counter = 0
      while not self.messages.empty() and self.currentTime and (self.currentTime <= self.stopTime):
        self.currentTime, event = self.messages.get()
        msg_recipient, msg_type, msg = event
        if msg_counter % 100000 == 0:
          print(f"[kernel] t={self.currentTime} processed={msg_counter}")
        msg_counter += 1
        if msg_type == MessageType.WAKEUP:
          target_id = msg_recipient
          agents[target_id].wakeup(self.currentTime)
        elif msg_type == MessageType.MESSAGE:
          target_id = msg_recipient
          agents[target_id].receiveMessage(self.currentTime, msg)
        else:
          raise ValueError("Unknown message type found in queue", "currentTime:", self.currentTime, "messageType:", self.msg.type)
      wall_stop = pd.Timestamp('now')
      elapsed_wall = wall_stop - wall_start
      for agent in agents:
        agent.kernelStopping()
      for agent in agents:
        agent.kernelTerminating()
      rate = msg_counter / (elapsed_wall / (np.timedelta64(1, 's')))
      print(f"[kernel] queue_elapsed={elapsed_wall} messages={msg_counter} rate={rate:0.1f}/s")
    print("[kernel] simulation ending")
    return {}

  def dispatch(self, sender = None, recipient = None, msg = None, delay = 0):

    if sender is None:
      raise ValueError("sendMessage() called without valid sender ID",
                       "sender:", sender, "recipient:", recipient,
                       "msg:", msg)

    if recipient is None:
      raise ValueError("sendMessage() called without valid recipient ID",
                       "sender:", sender, "recipient:", recipient,
                       "msg:", msg)

    if msg is None:
      raise ValueError("sendMessage() called with message == None",
                       "sender:", sender, "recipient:", recipient,
                       "msg:", msg)

    dispatch_ts = self.currentTime
    self.messages.put((dispatch_ts, (recipient, MessageType.MESSAGE, msg)))

  def schedule_wake(self, sender = None, requestedTime = None):

    if requestedTime is None:
        requestedTime = self.currentTime

    if sender is None:
      raise ValueError("setWakeup() called without valid sender ID",
                       "sender:", sender, "requestedTime:", requestedTime)

    wake_ts = requestedTime
    self.messages.put((wake_ts, (sender, MessageType.WAKEUP, None)))

  def getAgentComputeDelay(self, sender=None):
    return 0

  def setAgentComputeDelay(self, sender=None, requestedDelay=None):
    return

  def delayAgent(self, sender=None, additionalDelay=None):
    return

  def archive_df (self, sender, dfLog, filename=None):

    if self.skip_log: return
    dest_dir = os.path.join(".", "log", self.log_dir)
    if filename:
      fname = "{}.bz2".format(filename)
    else:
      fname = "{}.bz2".format(self.agents[sender].name.replace(" ",""))
    if not os.path.exists(dest_dir):
      os.makedirs(dest_dir)
    dfLog.to_pickle(os.path.join(dest_dir, fname), compression='bz2')

  def appendSummaryLog (self, sender, eventType, event):

    self.summaryLog.append({ 'AgentID' : sender,
                             'AgentStrategy' : self.agents[sender].type,
                             'EventType' : eventType, 'Event' : event })

  def writeSummaryLog (self):
    dest_dir = os.path.join(".", "log", self.log_dir)
    fname = "summary_log.bz2"
    if not os.path.exists(dest_dir):
      os.makedirs(dest_dir)
    dfLog = pd.DataFrame(self.summaryLog)
    dfLog.to_pickle(os.path.join(dest_dir, fname), compression='bz2')

  def updateAgentState (self, agent_id, state):
    if 'agent_state' not in self.custom_state: self.custom_state['agent_state'] = {}
    self.custom_state['agent_state'][agent_id] = state
