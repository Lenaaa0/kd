"""Model definitions for teacher and student architectures."""
from .teacher import TransformerTeacher
from .student_cnn import CNNStudent
from .student_lr import LRStudent

__all__ = ["TransformerTeacher", "CNNStudent", "LRStudent"]
