# Mapping of symptom codes to descriptive names
symptom_mapping = {
    'GP01': 'Tooth Pain',
    'GP02': 'Gum Bleeding',
    'GP03': 'Swollen Gums',
    'GP04': 'Bad Breath',
    'GP05': 'Tooth Sensitivity',
    'GP06': 'Loose Teeth',
    'GP07': 'Receding Gums',
    'GP08': 'Toothache When Chewing',
    'GP09': 'Jaw Pain',
    'GP10': 'Mouth Sores',
    'GP11': 'Tooth Discoloration',
    'GP12': 'Pus Around Teeth/Gums',
    'GP13': 'Facial Swelling',
    'GP14': 'Fever',
    'GP15': 'Difficulty Swallowing',
    'GP16': 'Tooth Cavities',
    'GP17': 'Gum Redness',
    'GP18': 'Persistent Dry Mouth',
    'GP19': 'Metallic Taste',
    'GP20': 'Visible Holes in Teeth',
    'GP21': 'Tooth Abscess',
    'GP22': 'Gum Inflammation',
    'GP23': 'Tooth Fracture',
    'GP24': 'Tooth Mobility',
    'GP25': 'Persistent Headache',
    'GP26': 'Swollen Lymph Nodes',
    'GP27': 'Painful Tongue',
    'GP28': 'Burning Mouth Sensation'
}

# Replace the original names definition
names = list(symptom_mapping.values()) + ['Diagnosis']

# If you need to keep the original dataset loading working:
# Create a mapping from new names back to original codes for dataset loading
reverse_mapping = {v: k for k, v in symptom_mapping.items()}