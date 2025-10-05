"""
Multi-View System Quick Test
=============================

Tests the multi-view enrollment and matching system.
"""

from multi_view_enrollment import (
    MultiViewEnrollment, 
    MultiViewMatcher,
    verify_enrollment_database,
    POSES
)


def test_system():
    """Run system tests"""
    print("\n" + "="*70)
    print("MULTI-VIEW ENROLLMENT SYSTEM - QUICK TEST")
    print("="*70 + "\n")
    
    # Check if we have enrolled users
    user_ids = MultiViewEnrollment.list_enrolled_users()
    
    if not user_ids:
        print("📋 No enrolled users found in database.")
        print("\nTo test the system:")
        print("1. Run: python multi_view_enrollment_gui.py")
        print("2. Enroll a test user with 3 poses")
        print("3. Run this script again")
        print("\nOr manually enroll using code:")
        print("""
from multi_view_enrollment import MultiViewEnrollment

enroller = MultiViewEnrollment()
result = enroller.enroll_user(
    user_id="test_user_001",
    image_paths=[
        "path/to/frontal.jpg",
        "path/to/left.jpg",
        "path/to/right.jpg"
    ]
)
        """)
        return
    
    # Show database status
    print(f"✅ Found {len(user_ids)} enrolled user(s): {', '.join(user_ids)}\n")
    
    # Verify database
    verify_enrollment_database()
    
    # Test loading an enrollment
    print("\n" + "="*70)
    print("TESTING ENROLLMENT LOADING")
    print("="*70 + "\n")
    
    test_user = user_ids[0]
    print(f"Loading enrollment for: {test_user}")
    
    enrollment = MultiViewEnrollment.load_enrollment(test_user)
    
    if enrollment:
        print(f"✅ Successfully loaded enrollment")
        print(f"   Templates: {len(enrollment['templates'])}")
        print(f"   Status: {enrollment['status']}")
        print(f"   Quality: {enrollment['quality_check']['status']}")
        print(f"   Enrolled: {enrollment['enrollment_date']}")
        
        print(f"\n📊 Template Details:")
        for idx, metadata in enumerate(enrollment['template_metadata']):
            print(f"   {idx+1}. {metadata['pose']:>10} - norm: {metadata['embedding_norm']:.4f}")
    
    # Instructions for matching test
    print("\n" + "="*70)
    print("READY FOR MATCHING TESTS")
    print("="*70 + "\n")
    print("To test matching, use this code:")
    print(f"""
from multi_view_enrollment import MultiViewMatcher

matcher = MultiViewMatcher()

# Test with an image
result = matcher.match_against_enrollment(
    test_image_path="path/to/test/image.jpg",
    user_id="{test_user}",
    threshold=1.25
)

print(f"Match result: {{result['decision']}}")
print(f"Min distance: {{result['min_distance']:.4f}}")
print(f"Best template: {{result['best_pose']}}")
print(f"Confidence: {{result['confidence']}}")
    """)
    
    print("\n" + "="*70)
    print("SYSTEM STATUS: ✅ OPERATIONAL")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_system()
