from .pages_features import feature_total_pages_count, feature_total_pages_count_1, feature_page_dynamic
from .level_features import last_level_of_each_customer
from .status import num_status, num_status_1
from .device_used import exact_device_type, exact_device_type_1
from .gender_feature import gender_feature_ont_hot, gender_feature_ont_hot_1
try:
    from .location_features import location_state_one_hot_encoded, location_state_one_hot_encoded_1
    _LOCATION_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency (skrub/matplotlib) may be missing
    location_state_one_hot_encoded = None
    location_state_one_hot_encoded_1 = None
    _LOCATION_AVAILABLE = False
from .registration_feature import member_since, member_since_1
from .song_artists_features import num_different_songs, num_different_artists, num_different_songs_1, num_different_artists_1
from .utils import weekly_features, rolling_mean_per_userId
from .time_features import compute_time_features


__all__ = [
    "feature_total_pages_count",
    "feature_total_pages_count_1",
    "feature_page_dynamic",
    "last_level_of_each_customer",
    "num_status",
    "num_status_1",
    "exact_device_type",
    "exact_device_type_1",
    "gender_feature_ont_hot",
    "gender_feature_ont_hot_1",
    "member_since",
    "member_since_1",
    "num_different_songs",
    "num_different_artists",
    "num_different_songs_1",
    "num_different_artists_1",
    "weekly_features",
    "rolling_mean_per_userId",
    "compute_time_features",
]

# If location dependencies are available, expose them; otherwise hide to avoid import errors.
if _LOCATION_AVAILABLE:
    __all__.extend(
        [
            "location_state_one_hot_encoded",
            "location_state_one_hot_encoded_1",
        ]
    )
