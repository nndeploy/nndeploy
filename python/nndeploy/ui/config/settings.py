"""
Global Settings Module

Responsibilities:
- Manage application-wide configuration parameters
- Provide interfaces for reading and writing configurations
- Handle configuration persistence
- Support configuration import and export

Configuration items are defined as dictionaries or classes, with support for loading from config files
"""

from typing import Any, Dict, Optional, Callable, List, Union
import json
import os
from pathlib import Path

class Settings:
    """
    Global Settings Manager
    
    Manages all application configuration parameters, provides unified read/write interfaces,
    and supports configuration persistence, import/export, and change notification mechanisms.
    """
    
    # Default settings
    DEFAULTS = {
        # Canvas settings
        "canvas": {
            "width": 1920,          # Default width
            "height": 1080,         # Default height
            "grid_size": 20,        # Grid size
            "grid_enabled": True,   # Show grid
            "grid_color": "#E5E5E5",# Grid color
            "snap_to_grid": True,   # Snap to grid
            "zoom_min": 0.1,        # Minimum zoom level
            "zoom_max": 5.0,        # Maximum zoom level
            "zoom_step": 0.1,       # Zoom step increment
        },
        
        # Node settings
        "node": {
            "default_width": 200,   # Default width
            "default_height": 100,  # Default height
            "min_width": 100,       # Minimum width
            "min_height": 50,       # Minimum height
            "padding": 10,          # Internal padding
            "border_radius": 6,     # Border radius
            "font_size": 14,        # Font size
            "line_height": 1.5,     # Line height
        },
        
        # Edge settings
        "edge": {
            "line_width": 2,        # Line width
            "line_color": "#666666",# Line color
            # "arrow_size": 8,        # Arrow size
            "curve_factor": 0.5,    # Curve factor
            "snap_distance": 10,    # Snap distance
        },
        
        # Auto-save settings
        "auto_save": {
            "enabled": True,        # Enable auto-save
            "interval": 300,        # Interval (seconds)
            "max_backups": 5,       # Maximum backup count
        },
        
        # Performance settings
        "performance": {
            "render_quality": "high",    # Render quality (low/medium/high)
            "animation_enabled": True,    # Enable animations
            "cache_size": 100,           # Cache size (MB)
            "max_undo_steps": 50,        # Maximum undo steps
        },
        
        # UI settings
        "ui": {
            "sidebar_width": 300,        # Sidebar width
            "panel_width": 400,          # Panel width
            "toolbar_position": "top",   # Toolbar position
            "show_status_bar": True,     # Show status bar
            "show_minimap": True,        # Show minimap
            "theme": "light",            # Theme mode (light/dark)
        }
    }
    
    def __init__(self):
        """Initialize settings manager, load defaults and update from config file"""
        self._settings: Dict[str, Dict[str, Any]] = self.DEFAULTS.copy()
        self._observers: List[Callable[[Optional[str], Optional[str], Any], None]] = []
        self._load_settings()
        
    def _load_settings(self) -> None:
        """Load settings from config file, keep defaults if file doesn't exist or has format errors"""
        config_path = self._get_config_path()
        
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved_settings = json.load(f)
                    # Recursively update settings while preserving default structure
                    self._update_dict(self._settings, saved_settings)
            except json.JSONDecodeError as e:
                print(f"Config file format error: {e}")
            except Exception as e:
                print(f"Failed to load settings: {e}")
                
    def _save_settings(self) -> None:
        """Save settings to config file, ensuring directory exists"""
        config_path = self._get_config_path()
        
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save settings: {e}")
            
    def _get_config_path(self) -> Path:
        """Get configuration file path"""
        # Use more reliable path construction
        return Path(os.path.dirname(os.path.abspath(__file__))) / "../config/settings.json"
        
    def _update_dict(self, target: Dict, source: Dict) -> None:
        """
        Recursively update dictionary while preserving target structure
        
        Args:
            target: Target dictionary
            source: Source dictionary
        """
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    self._update_dict(target[key], value)
                else:
                    target[key] = value
                    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get setting value
        
        Args:
            section: Setting section
            key: Setting key
            default: Default value
            
        Returns:
            Setting value, or default if not found
        """
        try:
            return self._settings[section][key]
        except KeyError:
            return default
            
    def set(self, section: str, key: str, value: Any, save: bool = True) -> bool:
        """
        Set setting value
        
        Args:
            section: Setting section
            key: Setting key
            value: Setting value
            save: Whether to save to file
            
        Returns:
            Whether setting was successfully updated
        """
        try:
            if section not in self._settings:
                self._settings[section] = {}
                
            if self._settings[section].get(key) != value:
                self._settings[section][key] = value
                # Notify observers
                for observer in self._observers:
                    observer(section, key, value)
                # Save to file
                if save:
                    self._save_settings()
                return True
            return False
        except Exception:
            return False
            
    def get_section(self, section: str) -> Dict:
        """
        Get entire section of settings
        
        Args:
            section: Setting section
            
        Returns:
            Copy of section settings dictionary
        """
        return self._settings.get(section, {}).copy()
        
    def reset_section(self, section: str, save: bool = True) -> bool:
        """
        Reset section settings to defaults
        
        Args:
            section: Setting section
            save: Whether to save to file
            
        Returns:
            Whether reset was successful
        """
        if section in self._settings and section in self.DEFAULTS:
            self._settings[section] = self.DEFAULTS[section].copy()
            # Notify observers
            for observer in self._observers:
                observer(section, None, self._settings[section])
            # Save to file
            if save:
                self._save_settings()
            return True
        return False
                
    def reset_all(self, save: bool = True) -> None:
        """
        Reset all settings to defaults
        
        Args:
            save: Whether to save to file
        """
        self._settings = self.DEFAULTS.copy()
        # Notify observers
        for observer in self._observers:
            observer(None, None, self._settings)
        # Save to file
        if save:
            self._save_settings()
            
    def add_observer(self, observer: Callable[[Optional[str], Optional[str], Any], None]) -> None:
        """
        Add settings change observer
        
        Args:
            observer: Observer callback function that receives (section, key, value) parameters
        """
        if observer not in self._observers:
            self._observers.append(observer)
            
    def remove_observer(self, observer: Callable[[Optional[str], Optional[str], Any], None]) -> None:
        """
        Remove settings change observer
        
        Args:
            observer: Observer callback function to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)
            
    def export_settings(self, filepath: Union[str, Path]) -> bool:
        """
        Export settings to file
        
        Args:
            filepath: Export file path
            
        Returns:
            Whether export was successful
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to export settings: {e}")
            return False
            
    def import_settings(self, filepath: Union[str, Path], save: bool = True) -> bool:
        """
        Import settings from file
        
        Args:
            filepath: Import file path
            save: Whether to save to config file
            
        Returns:
            Whether import was successful
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                imported_settings = json.load(f)
                # Recursively update settings
                self._update_dict(self._settings, imported_settings)
                # Notify observers
                for observer in self._observers:
                    observer(None, None, self._settings)
                # Save to file
                if save:
                    self._save_settings()
                return True
        except json.JSONDecodeError:
            print(f"Import file format error: {filepath}")
            return False
        except Exception as e:
            print(f"Failed to import settings: {e}")
            return False

# Create global settings instance
settings = Settings()

# Convenience functions
def get_setting(section: str, key: str, default: Any = None) -> Any:
    """
    Convenience function to get setting value
    
    Args:
        section: Setting section
        key: Setting key
        default: Default value
        
    Returns:
        Setting value
    """
    return settings.get(section, key, default)

def set_setting(section: str, key: str, value: Any, save: bool = True) -> bool:
    """
    Convenience function to set setting value
    
    Args:
        section: Setting section
        key: Setting key
        value: Setting value
        save: Whether to save to file
        
    Returns:
        Whether setting was successfully updated
    """
    return settings.set(section, key, value, save)