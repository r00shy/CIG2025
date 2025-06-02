from vgc2.agent import BattlePolicy, SelectionPolicy, TeamBuildPolicy
from vgc2.competition import Competitor

from .doubleLBattlePolicy import DoubleLBattlePolicy
from .doubleLSelectionPolicy import DoubleLSelectionPolicy
from .doubleLTeamBuildingPolicy import DoubleLTeamBuildPolicy

class DoubleLCompetitor(Competitor):
    """
    Author: Paul Lukas Lehmann, Edgar Lange
    """
    def __init__(self, name: str = "Double L"):
        self.__name = name
        self.__battle_policy = DoubleLBattlePolicy()
        self.__selection_policy = DoubleLSelectionPolicy()
        self.__team_build_policy = DoubleLTeamBuildPolicy()

    @property
    def battle_policy(self) -> BattlePolicy | None:
        return self.__battle_policy

    @property
    def selection_policy(self) -> SelectionPolicy | None:
        return self.__selection_policy

    @property
    def team_build_policy(self) -> TeamBuildPolicy | None:
        return self.__team_build_policy

    @property
    def name(self) -> str:
        return self.__name
