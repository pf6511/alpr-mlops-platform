
import re
from src.data.database import DatabaseManager

class AccessController:
    """
    Manages access control logic for the Parking System.
    """
    def __init__(self):
        self.db = DatabaseManager()
        self.allowlist = set()
        self.sync_from_database()

    def normalize(self, text):
        """
        Normalize plate text: uppercase, remove spaces/dashes/special chars.
        Examples: 
        - "51-A 77.529" -> "51A77529"
        - "ab-123-cd" -> "AB123CD"
        """
        if not text:
            return ""
        # Remove all non-alphanumeric characters
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def sync_from_database(self):
        """
        Refresh allowlist from database residents with acces='oui'.
        """
        self.allowlist.clear()
        try:
            authorized_plates = self.db.get_whitelist()
            for plate in authorized_plates:
                normalized = self.normalize(plate)
                if normalized:
                    self.allowlist.add(normalized)
            print(f"✅ AccessController synced: {len(self.allowlist)} authorized plates")
        except Exception as e:
            print(f"❌ Error syncing allowlist: {e}")

    def update(self, text_content):
        """
        Update allowlist from text content (one plate per line).
        NOTE: This legacy method now only updates the in-memory allowlist temporarily.
        For persistent changes, use the Administration tab.
        """
        self.allowlist.clear()
        # Resync from DB first to ensure we have the base
        self.sync_from_database()
        
        if not text_content:
            return self.get_list_as_text()
            
        # Add manual overrides (not saved to DB in this version)
        for line in text_content.split('\n'):
            line = line.strip()
            if line:
                normalized = self.normalize(line)
                if normalized:
                    self.allowlist.add(normalized)
        
        return f"```\n{self.get_list_as_text()}\n```"
                    
    def check_access(self, plate_text):
        """
        Check if a plate is authorized.
        Returns: (is_authorized, formatted_message)
        """
        if not plate_text:
            return False, "NO PLATE DETECTED"
            
        normalized_input = self.normalize(plate_text)
        
        # Auto-refresh if empty (e.g. restart)
        if not self.allowlist:
            self.sync_from_database()
            
        authorized = normalized_input in self.allowlist
        
        if authorized:
            return True, f"✅ ACCESS GRANTED: {plate_text}"
        else:
            return False, f"⛔ ACCESS DENIED: {plate_text}"

    def log_attempt(self, plate_text, authorized):
        """Log access attempt to SQLite database"""
        if not plate_text:
            return
            
        try:
            normalized = self.normalize(plate_text)
            self.db.add_log(plate_text, authorized, normalized)
        except Exception as e:
            print(f"Error logging access: {e}")

    def get_list_as_text(self):
        """Returns current allowlist formatted for text area"""
        return "\n".join(sorted(list(self.allowlist)))
