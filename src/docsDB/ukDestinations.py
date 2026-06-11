uk_destinations = [("Cornwall", "Cornwall"), 
                   ("North_Cornwall", "Cornwall"), 
                   ("South_Cornwall", "Cornwall"), 
                   ("West_Cornwall", "Cornwall"),
                   ("Tintagel", "Cornwall"), 
                   ("Bodmin", "Cornwall"), 
                   ("Wadebridge", "Cornwall"),
                   ("Penzance", "Cornwall"), 
                   ("Newquay", "Cornwall"), 
                   ("St_Ives", "Cornwall"),
                   ("Port_Isaac", "Cornwall"), 
                   ("Looe", "Cornwall"), 
                   ("Polperro", "Cornwall"),
                   ("Porthleven", "Cornwall"),
                   ("East_Sussex", "East_Sussex"),
                   ("Brighton", "East_Sussex"),
                   ("Battle", "East_Sussex"),
                   ("Hastings_(England)", "East_Sussex"),
                   ("Rye_(England)", "East_Sussex"), 
                   ("Seaford", "East_Sussex"), 
                   ("Ashdown_Forest", "East_Sussex")]

wikivoyage_root_url = "https://en.wikivoyage.org/wiki"

def getDestinations_with_metadata():
  return [(f'{wikivoyage_root_url}/{destination}', destination, region) for destination, region in uk_destinations]
