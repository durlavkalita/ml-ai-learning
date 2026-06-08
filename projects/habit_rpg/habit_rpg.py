import os
import random
from typing import List, Dict
from pydantic import BaseModel, Field

class PlayerModel(BaseModel):
    name: str = 'Hero'
    level: int = 1
    xp: int = 0
    hp: int = 100
    inventory: List[str] = Field(default_factory=list)

    @property
    def xp_needed(self) -> int:
        return self.level * 100

class HabitModel(BaseModel):
    streak: int = 0

class GameStateModel(BaseModel):
    player: PlayerModel = Field(default_factory=PlayerModel)
    habits: Dict[str, HabitModel] = Field(default_factory=dict)

class GameStateRepository:
    def __init__(self, filepath: str = "rpg_data.json"):
        self.filepath = filepath

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                return GameStateModel.model_validate_json(f.read())
        return GameStateModel()
            
    def save(self, state: GameStateModel):
        with open(self.filepath, 'w') as f:
            f.write(state.model_dump_json(indent=4))

ITEMS_POOL = ["Iron Sword", "Dragon Shield", "Health Potion", "Wizard Hat", "Shiny Coin", "Phoenix Down"]

class GameEngine:
    def __init__(self, repo: GameStateRepository):
        self.repo = repo
        self.state = self.repo.load()

    def _check_level_up(self) -> None:
        player = self.state.player
        if player.xp >= player.xp_needed:
            player.xp -= player.xp_needed
            player.level += 1
            player.hp = 100 # heal on new level
            print(f"\nLEVEL UP! You are now Level {player.level}!")

    def complete_habit(self, habit_name: str) -> None:
        if habit_name not in self.state.habits:
            print("Habit not found!")
            return
        
        habit = self.state.habits[habit_name]
        habit.streak += 1
        xp_gained = 20 + (habit.streak * 5) #streak bonus
        self.state.player.xp += xp_gained

        print(f"\nConquered : {habit_name}!")
        print(f"Gained {xp_gained} XP.")

        # 30% chance of loot drop
        if random.random() < 0.3:
            item = random.choice(ITEMS_POOL)
            self.state.player.inventory.append(item)
            print(f"Loot Drop: You found [{item}]!")

        self._check_level_up()
        self.repo.save(self.state)

    def fail_habit(self, habit_name: str) -> None:
        if habit_name not in self.state.habits:
            print("Habit not found!")
            return
        
        habit = self.state.habits[habit_name]
        habit.streak = 0
        damage = 15
        self.state.player.hp -= damage
        print(f"\nFailed: {habit_name}! You took {damage} damage.")

        if self.state.player.hp <= 0:
            print("You died. Your gold and items are lost, HP reset to 50")
            self.state.player.hp = 50
            self.state.player.inventory = []

        self.repo.save(self.state)

    def add_habit(self, habit_name: str) -> None:
        if habit_name and habit_name not in self.state.habits:
            self.state.habits[habit_name] = HabitModel()
            self.repo.save(self.state)

    def show_dashboard(self) -> None:
        p = self.state.player
        print("\n" + "="*40)
        print(f"🧙 {p.name} | Lv. {p.level} | HP: {p.hp}/100 | XP: {p.xp}/{p.level*100}")
        print(f"🎒 Inventory: {', '.join(p.inventory) if p.inventory else 'Empty'}")
        print("="*40)
        print("\n⚡ YOUR QUESTS (HABITS):")
        if not self.state.habits:
            print("   No quests accepted yet. Add a habit!")
        for name, info in self.state.habits.items():
            print(f"   - {name} (🔥 Streak: {info.streak})")
        print("="*40)

def main():
    repo = GameStateRepository()
    game = GameEngine(repo)
    
    while True:
        game.show_dashboard()
        print("\n[1] Add New Quest (Habit)")
        print("[2] Complete Quest (Gain XP & Loot)")
        print("[3] Fail Quest (Take Damage)")
        print("[4] Rage Quit")
        
        choice = input("\nWhat will you do? ").strip()
        
        if choice == "1":
            name = input("Enter quest name: ").strip()
            if name:
                game.add_habit(name)
        elif choice == "2":
            name = input("Which quest did you smash? ").strip()
            game.complete_habit(name)
        elif choice == "3":
            name = input("Which quest defeated you? ").strip()
            game.fail_habit(name)
        elif choice == "4":
            print("Goodbye, Adventurer!")
            break
        else:
            print("Invalid action!")

if __name__ == "__main__":
    main()