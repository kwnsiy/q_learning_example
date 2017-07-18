# coding: utf-8

using DataFrames
include("rl_example.jl")

doc=
"""
逆強化学習: [Abbeel et al. 2004]
適当
"""

# gamma 
gamma = 0.8

# state
n_state = 6

# action 
n_action = 6

# エキスパートの行動
expert_log = eg_rl_grid()
state_log = [x[1] for x in expert_log]
action_log =　[x[2] for x in expert_log]

# 状態遷移関数
tmp = [ -1 -1 -1 -1 0 -1;
  -1 -1 -1 0 -1 0;
  -1 -1 -1 0 -1 -1;
  -1  0 0 -1  0 -1;
  0 -1 -1  0 -1 0;
  -1  0 -1 -1  0 0]
D =  Dict(s => find(tmp[s, :] .!= -1) for s in collect(1:6))

# feature 
f = tmp + 1

state_log = action_log

# init u0 & ue 
ue = sum(map(x -> sum([(gamma^i) * f[s, :] for (i, s) in enumerate(x)]), state_log)) / length(state_log)''
u = [mean([(gamma^i) * f[s, :] for (i, s) in enumerate(state_log[1])])'']

# i = 1
mu =  [u[end]]
w = [ue - mu[end]]
t = [norm(w[end])]

Q = zeros(6, 6)

@printf " ---------- irl ---------- \n"

for demo_i in collect(1:100)  
  # 閾値
　 @show t[end]

  # 報酬関数
  #R = w[end]' .* f
  R = (sum(w)/sum(sum(w))) .* f
  @show R
  
  #行動価値関数
　 Q = zeros(6, 6)

  q_learn!(Q, R, D, gamma, 1000, 6)  
  # best policy
  policy = run_greedy(Q, R, D, rand(1:n_state), 6, false)
  policy = [p for p in policy[2] if p == 6]  
  push!(u, mean([(gamma^i) * f[s, :] for (i, s) in enumerate(policy[rand(1:end)])])'')
  
  println("Q function = ")
  println(DataFrame(Q))
  println("R function = ")
  println(DataFrame(R))
  
  # update
  push!(mu, mu[end] .+ ( ((u[end] - mu[end])' * (ue - mu[end])) / ((u[end] - mu[end])' * (u[end] - mu[end])) ) .* (u[end] - mu[end]))
  push!(w, ue - mu[end])
  push!(t, norm(w[end]))
  # stop
  # abs(t[end - 1] - t[end]) <= 0.00000000000001 && break
end

# 逆強化学習で得られた報酬関数を用いてQ学習
R = (sum(w)/sum(sum(w)))' .* f
Q = zeros(6, 6)
q_learn!(Q, R, D, gamma, 1000, 6)
@show run_greedy(Q, R, D, 1, 6, false)
@show run_greedy(Q, R, D, 2, 6, false)
@show run_greedy(Q, R, D, 3, 6, false)
@show run_greedy(Q, R, D, 4, 6, false)
@show run_greedy(Q, R, D, 5, 6, false)
@show run_greedy(Q, R, D, 6, 6, false)

Q = Q / maximum(Q) * 100
R = R / maximum(R) * 100

println("Q function = ")
println(DataFrame(Q))
println("R function = ")
println(DataFrame(R))
