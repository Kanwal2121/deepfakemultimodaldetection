import tensorflow as tf
from model import create_visual_expert
from model_audio import create_audio_expert

print("Creating dummy Visual Expert...")
vis_model = create_visual_expert()
vis_model.save("visual_expert_best.keras")
print("Saved visual_expert_best.keras")

print("Creating dummy Audio Expert...")
aud_model = create_audio_expert()
aud_model.save("audio_expert_best.keras")
print("Saved audio_expert_best.keras")
