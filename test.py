import sys

def solve():
    t = int(input())
    res = []

    for _ in range(t):
        n = int(input())

        a = []
        while len(a) < n:
            a.extend(map(int, input().split()))

        INF = 10 ** 18
        dp = [0] * (n + 1)

        # 因为题目说 0 <= ai <= n，所以开 n + 1 大小即可
        best = [INF] * (n + 1)

        for i in range(1, n + 1):
            x = a[i - 1]

            # 情况1：保留当前数字
            dp[i] = dp[i - 1] + 1

            # 情况2：当前数字和前面相同数字配对消除
            if best[x] < dp[i]:
                dp[i] = best[x]

            # 当前数字可以作为以后消除的左端点
            if dp[i - 1] < best[x]:
                best[x] = dp[i - 1]

        res.append(str(dp[n]))

    print("\n".join(res))


if __name__ == "__main__":
    solve()