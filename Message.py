from enum import Enum, unique

@unique
class MessageType(Enum):
  MESSAGE = 1
  WAKEUP = 2

  def __lt__(self, other):
    return self.value < other.value


class Message:

  uniq = 0

  def __init__(self, body=None):
    self.body = body
    self.uniq = Message.uniq
    Message.uniq += 1

  def __lt__(self, other):
    return self.uniq < other.uniq

  def __str__(self):
    return str(self.body)
