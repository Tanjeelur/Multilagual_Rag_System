import redis
r = redis.Redis(
    host="fitting-puma-60812.upstash.io",
    port=6379,
    password="Ae2MAAIjcDEzNzYzNDU1ODBmNGU0NWY2ODQ4MGQxYWY1NTExOWVhN3AxMA",
    ssl=True
)
print(r.ping())  # Should print: True