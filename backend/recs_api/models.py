import secrets
from django.db import models
from django.contrib.auth.models import User


class AuthToken(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='auth_token')
    key = models.CharField(max_length=64, unique=True)

    @staticmethod
    def generate(user):
        AuthToken.objects.filter(user=user).delete()
        token = AuthToken.objects.create(user=user, key=secrets.token_hex(32))
        return token.key


class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='search_history')
    query = models.CharField(max_length=500)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username}: {self.query}"
