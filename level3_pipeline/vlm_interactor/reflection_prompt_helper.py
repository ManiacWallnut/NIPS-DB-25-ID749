from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import glog

"""


This file contains the definition of the ReflectionPromptHelper class, which is responsible for

generating and managing reflection prompts for a given task. The class includes methods for
creating reflection prompts, adding content to the conversation, and handling user input."""
