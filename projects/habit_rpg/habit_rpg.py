import json
import os
import random

DATA_FILE = "rpg_data.json"

def load_data()->dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {
        "player": {
            "name": "Hero",
            "level": 1,
            "xp": 0,
            "hp": 100,
            "inventory": []
        },
        "habits": {}
    }

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

ITEMS_POOL = ["Iron Sword", "Dragon Shield", "Health Potion", "Wizard Hat", "Shiny Coin", "Phoenix Down"]

def check_level_up(player):
    xp_needed = player["level"]*100
    if player["xp"] >= xp_needed:
        player["xp"] -= xp_needed
        player["level"] += 1
        player["hp"] = 100 # heal on new level
        print(f"\nLEVEL UP! You are now Level {player['level']}!")

def complete_habit(data, habit_name):
    if habit_name not in data['habits']:
        print("Habit not found")
        return
    
    data["habits"][habit_name]["streak"] += 1
    xp_gained = 20 + (data["habits"][habit_name]["streak"] * 5) #streak bonus
    data["player"]["xp"] += xp_gained

    print(f"\nConquered : {habit_name}!")
    print(f"Gained {xp_gained} XP.")

    # 30% chance of loot drop
    if random.random() < 0.3:
        item = random.choice(ITEMS_POOL)
        data["player"]["inventory"].append(item)
        print(f"Loot Drop: You found [{item}]!")

    check_level_up(data['player'])
    save_data(data)

def fail_habit(data, habit_name):
    if habit_name not in data['habits']:
        print("Habit not found!")
        return
    
    data["habits"][habit_name]["streak"] = 0
    damage = 15
    data["player"]["hp"] -= damage
    print(f"\nFailed: {habit_name}! You took {damage} damage.")

    if data['player']['hp'] <= 0:
        print("You died. Your gold and items are lost, HP reset to 50")
        data["player"]["hp"] = 50
        data["player"]["inventory"] = []

    save_data(data)

def show_dashboard(data):
    p = data["player"]
    print("\n" + "="*40)
    print(f"🧙 {p['name']} | Lv. {p['level']} | HP: {p['hp']}/100 | XP: {p['xp']}/{p['level']*100}")
    print(f"🎒 Inventory: {', '.join(p['inventory']) if p['inventory'] else 'Empty'}")
    print("="*40)
    print("\n⚡ YOUR QUESTS (HABITS):")
    if not data["habits"]:
        print("   No quests accepted yet. Add a habit!")
    for name, info in data["habits"].items():
        print(f"   - {name} (🔥 Streak: {info['streak']})")
    print("="*40)

def main():
    data = load_data()
    
    while True:
        show_dashboard(data)
        print("\n[1] Add New Quest (Habit)")
        print("[2] Complete Quest (Gain XP & Loot)")
        print("[3] Fail Quest (Take Damage)")
        print("[4] Rage Quit")
        
        choice = input("\nWhat will you do? ").strip()
        
        if choice == "1":
            name = input("Enter quest name: ").strip()
            if name:
                data["habits"][name] = {"streak": 0}
                save_data(data)
        elif choice == "2":
            name = input("Which quest did you smash? ").strip()
            complete_habit(data, name)
        elif choice == "3":
            name = input("Which quest defeated you? ").strip()
            fail_habit(data, name)
        elif choice == "4":
            print("Goodbye, Adventurer!")
            break
        else:
            print("Invalid action!")

if __name__ == "__main__":
    main()