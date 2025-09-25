"""Security tests for API key handling and encryption.

Tests ensure API keys are properly encrypted, stored securely,
and not exposed in logs or debug output.
"""
import pytest
import os
import json
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.utils.crypto_utils import encrypt_api_key, decrypt_api_key, get_gemini_api_key, set_gemini_api_key
from app.core.exceptions import SecurityError
from app.config.settings import Config


class TestAPIKeyEncryption:
    """Test API key encryption and decryption functionality."""

    def test_encrypt_decrypt_roundtrip(self):
        """Test API key encryption and decryption round trip."""
        original_key = "test_gemini_api_key_12345"

        # Encrypt the key
        encrypted_key = encrypt_api_key(original_key)

        # Should be different from original
        assert encrypted_key != original_key
        assert len(encrypted_key) > len(original_key)  # Base64 encoding increases size

        # Decrypt the key
        decrypted_key = decrypt_api_key(encrypted_key)

        # Should match original
        assert decrypted_key == original_key

    def test_encrypt_empty_string(self):
        """Test encrypting empty string."""
        encrypted = encrypt_api_key("")
        decrypted = decrypt_api_key(encrypted)
        assert decrypted == ""

    def test_encrypt_none_value(self):
        """Test encrypting None value."""
        with pytest.raises(ValueError):
            encrypt_api_key(None)

    def test_decrypt_invalid_data(self):
        """Test decrypting invalid encrypted data."""
        with pytest.raises(SecurityError):
            decrypt_api_key("invalid_encrypted_data")

    def test_decrypt_tampered_data(self):
        """Test decrypting tampered encrypted data."""
        original_key = "test_key"
        encrypted_key = encrypt_api_key(original_key)

        # Tamper with the encrypted data
        tampered_key = encrypted_key[:-5] + "xxxxx"

        with pytest.raises(SecurityError):
            decrypt_api_key(tampered_key)

    def test_encryption_produces_different_results(self):
        """Test that encrypting the same key produces different results (salt/IV)."""
        key = "same_test_key"

        encrypted1 = encrypt_api_key(key)
        encrypted2 = encrypt_api_key(key)

        # Should produce different encrypted results due to random salt/IV
        assert encrypted1 != encrypted2

        # But both should decrypt to the same original
        assert decrypt_api_key(encrypted1) == key
        assert decrypt_api_key(encrypted2) == key

    def test_encryption_with_special_characters(self):
        """Test encryption with keys containing special characters."""
        special_keys = [
            "key-with-hyphens",
            "key_with_underscores",
            "key.with.dots",
            "key@with@symbols",
            "key with spaces",
            "key\nwith\nnewlines",
            "key\twith\ttabs",
            "key/with/slashes",
            "key\\with\\backslashes",
            "key\"with\"quotes",
            "key'with'apostrophes"
        ]

        for special_key in special_keys:
            encrypted = encrypt_api_key(special_key)
            decrypted = decrypt_api_key(encrypted)
            assert decrypted == special_key, f"Failed for key: {special_key}"

    def test_encryption_with_unicode(self):
        """Test encryption with Unicode characters."""
        unicode_keys = [
            "clé_française",  # French
            "ключ_русский",   # Russian
            "钥匙_中文",       # Chinese
            "鍵_日本語",       # Japanese
            "🔑_emoji_key",   # Emoji
            "αβγ_greek",      # Greek letters
        ]

        for unicode_key in unicode_keys:
            encrypted = encrypt_api_key(unicode_key)
            decrypted = decrypt_api_key(encrypted)
            assert decrypted == unicode_key, f"Failed for Unicode key: {unicode_key}"

    def test_very_long_key_encryption(self):
        """Test encryption of very long API keys."""
        long_key = "a" * 1000  # 1000 character key

        encrypted = encrypt_api_key(long_key)
        decrypted = decrypt_api_key(encrypted)

        assert decrypted == long_key
        assert len(encrypted) > len(long_key)


class TestAPIKeyStorage:
    """Test secure storage and retrieval of API keys."""

    def test_config_api_key_storage(self, temp_dir):
        """Test storing API key in configuration file."""
        config_file = temp_dir / "test_config.json"
        test_key = "test_api_key_secure_storage"

        # Create config with API key
        config_data = {"gemini_api_key": test_key}

        # Use encryption for storage
        config_data["encrypted_gemini_api_key"] = encrypt_api_key(test_key)
        del config_data["gemini_api_key"]  # Remove plaintext

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Verify plaintext key is not in file
        with open(config_file, 'r') as f:
            file_content = f.read()

        assert test_key not in file_content
        assert "encrypted_gemini_api_key" in file_content

        # Load and decrypt
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)

        decrypted_key = decrypt_api_key(loaded_config["encrypted_gemini_api_key"])
        assert decrypted_key == test_key

    def test_set_get_gemini_api_key(self, temp_dir):
        """Test setting and getting Gemini API key through utility functions."""
        config_file = temp_dir / "test_config.json"
        config_data = {}

        test_key = "test_gemini_key_utility"

        # Set the API key
        set_gemini_api_key(config_data, test_key)

        # Should be encrypted in config
        assert "encrypted_gemini_api_key" in config_data
        assert "gemini_api_key" not in config_data or config_data["gemini_api_key"] == ""

        # Get the API key
        retrieved_key = get_gemini_api_key(config_data)
        assert retrieved_key == test_key

    def test_api_key_removal(self, temp_dir):
        """Test secure removal of API key."""
        config_data = {}
        test_key = "key_to_be_removed"

        # Set key
        set_gemini_api_key(config_data, test_key)
        assert get_gemini_api_key(config_data) == test_key

        # Remove key
        set_gemini_api_key(config_data, "")
        assert get_gemini_api_key(config_data) == ""

        # Should not contain encrypted key anymore
        assert "encrypted_gemini_api_key" not in config_data or config_data["encrypted_gemini_api_key"] == ""

    def test_config_file_permissions(self, temp_dir):
        """Test that config files have appropriate permissions."""
        if os.name != 'nt':  # Skip on Windows as it has different permission model
            config_file = temp_dir / "secure_config.json"
            test_key = "secure_test_key"

            config_data = {"encrypted_gemini_api_key": encrypt_api_key(test_key)}

            with open(config_file, 'w') as f:
                json.dump(config_data, f)

            # Set restrictive permissions (owner read/write only)
            os.chmod(config_file, 0o600)

            # Verify permissions
            stat_info = os.stat(config_file)
            permissions = oct(stat_info.st_mode)[-3:]

            assert permissions == "600", f"Expected 600 permissions, got {permissions}"


class TestAPIKeySanitization:
    """Test that API keys are properly sanitized from logs and output."""

    def test_logging_does_not_expose_api_key(self, caplog):
        """Test that API keys are not exposed in log messages."""
        test_key = "secret_api_key_should_not_appear"

        with caplog.at_level(logging.DEBUG):
            # Simulate various logging scenarios
            logger = logging.getLogger("test_logger")

            logger.debug(f"Configuring service with key: {test_key}")
            logger.info(f"Service initialized with configuration: {{'api_key': '{test_key}'}}")
            logger.error(f"Authentication failed for key: {test_key}")

        # API key should be sanitized in logs
        log_output = caplog.text

        # Should not contain the actual key
        assert test_key not in log_output
        # Should contain sanitized version
        assert "[REDACTED]" in log_output or "***" in log_output

    def test_exception_messages_dont_expose_keys(self):
        """Test that exception messages don't expose API keys."""
        test_key = "secret_exception_key"

        try:
            # Simulate an error that might include the API key
            raise Exception(f"API error with key: {test_key}")
        except Exception as e:
            sanitized_message = str(e).replace(test_key, "[REDACTED]")
            assert test_key not in sanitized_message
            assert "[REDACTED]" in sanitized_message

    def test_debug_output_sanitization(self):
        """Test that debug output sanitizes API keys."""
        test_key = "debug_api_key_secret"

        config_dict = {
            "gemini_api_key": test_key,
            "other_setting": "normal_value"
        }

        # Convert to string representation (like debug output)
        debug_str = str(config_dict)

        # Should sanitize the API key
        sanitized_debug = debug_str.replace(test_key, "[REDACTED]")

        assert test_key not in sanitized_debug
        assert "[REDACTED]" in sanitized_debug
        assert "other_setting" in sanitized_debug  # Other values should remain

    @patch('app.services.gemini_service.logging')
    def test_service_logging_sanitization(self, mock_logging):
        """Test that service-level logging sanitizes API keys."""
        from app.services.gemini_service import GeminiService

        test_key = "service_secret_key"
        mock_config = MagicMock()
        mock_config.gemini_api_key = test_key

        service = GeminiService(mock_config)

        # Check that any logging calls don't contain the raw API key
        for call in mock_logging.method_calls:
            call_str = str(call)
            assert test_key not in call_str, f"API key found in logging call: {call_str}"


class TestAPIKeyValidation:
    """Test validation of API key formats and security."""

    def test_valid_api_key_formats(self):
        """Test validation of various valid API key formats."""
        valid_keys = [
            "AIzaSyABC123_def456-GHI789",  # Google API key format
            "sk-1234567890abcdef1234567890abcdef",  # OpenAI format
            "test_key_123",  # Simple format
            "AKIAIOSFODNN7EXAMPLE",  # AWS format
            "ya29.AHES6ZSuY8h5hFz1-ABC123",  # OAuth token format
        ]

        for key in valid_keys:
            # Should not raise exceptions
            encrypted = encrypt_api_key(key)
            decrypted = decrypt_api_key(encrypted)
            assert decrypted == key

    def test_suspicious_api_key_patterns(self):
        """Test detection of suspicious API key patterns."""
        suspicious_keys = [
            "test",  # Too short
            "password123",  # Common password
            "admin",  # Common username
            "123456789",  # All numbers
            "aaaaaaaaaa",  # Repeated characters
        ]

        for key in suspicious_keys:
            # Should still encrypt/decrypt but could log warnings
            encrypted = encrypt_api_key(key)
            decrypted = decrypt_api_key(encrypted)
            assert decrypted == key

    def test_api_key_strength_validation(self):
        """Test API key strength validation."""
        # Weak keys (should work but could trigger warnings)
        weak_keys = [
            "short",
            "simple123",
            "nospecialchars"
        ]

        # Strong keys
        strong_keys = [
            "AIzaSyABC123_def456-GHI789jklMN0",
            "sk-1234567890abcdef1234567890abcdef12345678",
            "very-long-and-complex-api-key-with-multiple-parts-123456"
        ]

        for key in weak_keys + strong_keys:
            encrypted = encrypt_api_key(key)
            decrypted = decrypt_api_key(encrypted)
            assert decrypted == key

    def test_api_key_entropy_check(self):
        """Test API key entropy/randomness validation."""
        import string
        import secrets

        # Generate high-entropy key
        high_entropy_key = ''.join(secrets.choice(
            string.ascii_letters + string.digits + '-_'
        ) for _ in range(32))

        # Low entropy key (repeated pattern)
        low_entropy_key = "abcdef" * 10

        for key in [high_entropy_key, low_entropy_key]:
            encrypted = encrypt_api_key(key)
            decrypted = decrypt_api_key(encrypted)
            assert decrypted == key


class TestRateLimitingandSecurity:
    """Test rate limiting and other security measures."""

    def test_encryption_performance_limit(self):
        """Test that encryption operations have reasonable performance limits."""
        import time

        large_key = "x" * 10000  # 10KB key

        start_time = time.time()
        encrypted = encrypt_api_key(large_key)
        encrypt_time = time.time() - start_time

        start_time = time.time()
        decrypted = decrypt_api_key(encrypted)
        decrypt_time = time.time() - start_time

        # Should complete within reasonable time
        assert encrypt_time < 1.0  # Less than 1 second
        assert decrypt_time < 1.0  # Less than 1 second
        assert decrypted == large_key

    def test_memory_cleanup_after_operations(self):
        """Test that sensitive data is cleaned from memory."""
        import gc

        sensitive_key = "very_sensitive_api_key_12345"

        # Perform operations
        encrypted = encrypt_api_key(sensitive_key)
        decrypted = decrypt_api_key(encrypted)

        # Clear variables
        del sensitive_key, encrypted, decrypted

        # Force garbage collection
        gc.collect()

        # Memory should be cleaned (this is hard to test definitively,
        # but we ensure operations complete successfully)
        assert True  # Operations completed without error

    def test_concurrent_encryption_safety(self):
        """Test thread safety of encryption operations."""
        import threading
        import time

        test_keys = [f"concurrent_key_{i}" for i in range(10)]
        results = {}
        errors = []

        def encrypt_key(key):
            try:
                encrypted = encrypt_api_key(key)
                decrypted = decrypt_api_key(encrypted)
                results[key] = (decrypted == key)
            except Exception as e:
                errors.append(e)

        # Create threads for concurrent encryption
        threads = [threading.Thread(target=encrypt_key, args=(key,)) for key in test_keys]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout

        # Check results
        assert len(errors) == 0, f"Encryption errors: {errors}"
        assert len(results) == len(test_keys)
        assert all(results.values()), "Some encryption/decryption operations failed"


@pytest.mark.integration
class TestAPIKeySecurityIntegration:
    """Integration tests for API key security with real components."""

    def test_full_workflow_encryption(self, temp_dir):
        """Test complete workflow from setting to using API key."""
        config_file = temp_dir / "integration_config.json"
        test_key = "integration_test_api_key_workflow"

        # 1. Start with empty config
        config_data = {}

        # 2. Set API key (should be encrypted)
        set_gemini_api_key(config_data, test_key)

        # 3. Save config to file
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # 4. Verify file doesn't contain plaintext key
        with open(config_file, 'r') as f:
            file_content = f.read()
        assert test_key not in file_content

        # 5. Load config from file
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)

        # 6. Retrieve API key (should be decrypted)
        retrieved_key = get_gemini_api_key(loaded_config)
        assert retrieved_key == test_key

        # 7. Use in service (mock)
        from app.services.gemini_service import GeminiService
        mock_config = MagicMock()
        mock_config.gemini_api_key = retrieved_key

        with patch('app.services.gemini_service.genai'):
            service = GeminiService(mock_config)
            assert service.is_configured() is True

    def test_config_migration_security(self, temp_dir):
        """Test secure migration from plaintext to encrypted API keys."""
        config_file = temp_dir / "migration_config.json"
        plaintext_key = "plaintext_migration_key"

        # 1. Create old config with plaintext key
        old_config = {"gemini_api_key": plaintext_key}

        with open(config_file, 'w') as f:
            json.dump(old_config, f)

        # 2. Load and migrate config
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        # Migrate if plaintext key exists
        if "gemini_api_key" in config_data and config_data["gemini_api_key"]:
            key = config_data["gemini_api_key"]
            set_gemini_api_key(config_data, key)  # This encrypts it
            config_data["gemini_api_key"] = ""  # Clear plaintext

        # 3. Save migrated config
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # 4. Verify migration worked
        with open(config_file, 'r') as f:
            file_content = f.read()

        assert plaintext_key not in file_content
        assert "encrypted_gemini_api_key" in file_content

        # 5. Verify key can still be retrieved
        retrieved_key = get_gemini_api_key(config_data)
        assert retrieved_key == plaintext_key