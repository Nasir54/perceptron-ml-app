"""Basic test file to verify imports work"""
def test_imports():
    """Test that main modules can be imported"""
    try:
        from src.models import *
        from src.utils import *
        from src.data import *
        assert True
    except ImportError:
        assert False, "Import failed"

def test_basic_math():
    """Simple test to verify testing works"""
    assert 1 + 1 == 2
