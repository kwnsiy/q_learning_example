# coding: utf-8

using DataFrames

doc=
"""
強化学習 & 逆強化学習
2017/07/18
"""

# ランダムな状態遷移
function get_random_state(num_state)
  return rand(1:num_state)
end    

# ランダムな行動選択
function get_random_action(num_action)
  return rand(1:num_action)
end

# 可能な状態遷移
function get_possible_actions(D, state, num_state)
  state < 1 && throw("invaid state: $state")
  D == nothing && return get_random_state(num_state)
  return collect(keys(D[state]))
end

# 可能な状態遷移の中で最も高いq値を選択
function get_maxq_from_possible_actions(Q, state, possible_actions)
  return maximum([Q[state][i] for i in possible_actions])
end

# 特徴ベクトルを用いて価値関数を更新
function update_q!(Q, f, w, num_state, num_action)
  for s in collect(1:num_state)
    for a in collect(1:num_action)
      Q[s][a] = dot(w[1]', f[s][a])
    end 
  end
end

# 特徴ベクトルを用いて報酬関数を更新
function update_r!(R, f, w, num_state, num_action)
  for s in collect(1:num_state)
    for a in collect(1:num_action)
      R[s][a] = dot(w[1]', f[s][a])
    end 
  end
end


# q学習
function q_learn!(Q, R, D, w, f, gamma, learn_count, goal_state, num_state, num_action)
  # 初期状態を定義
  state = get_random_state(num_state)
  # シミュレーター
  for i in collect(1:learn_count)
    # i%100 == 0 && @printf("ite = %d \n", i)
    # 可能な状態遷移
    possible_actions = get_possible_actions(D, state, num_state)    
    # ランダムに行動選択
    action = possible_actions[rand(1:end)]
    if D != nothing
      next_state = D[state][action]
    else
      next_state = get_random_state(num_state)
    end
    
    #update_q!(Q, f, w, num_state, num_action)
    Q[state][action] = dot(w[1]', f[state][action])
    next_possible_actions = get_possible_actions(D, next_state, num_state)
    max_q_next_s = get_maxq_from_possible_actions(Q, next_state, next_possible_actions)
    delta = R[state][action] + (gamma * max_q_next_s - Q[state][action])
    
    w[1] = w[1]+ 0.1 * (delta * f[state][action])
    
    state = next_state 
    # 終了判定
    if state == goal_state
      # 初期状態を変更して続行
      state = get_random_state(goal_state)
    end
  end
end

# 学習結果の確認
function run_greedy(Q, R, D, num_state, num_action, start_state, goal_state,　end_action, greedy)
  state_log = []
  action_log = []
  i, flg, state = 0, 0, start_state
  while flg != goal_state
    #@printf "current state: %d \n"  state    
    # 現在の状態において可能な行動を取得
    possible_actions = get_possible_actions(D, state, num_action)    
    # Q-value(s, a)を最大化する行動を選択
    max_Q = 0
    best_action_candidates = []
    for a in possible_actions
      if Q[state][a] > max_Q
        push!(best_action_candidates, a)
        max_Q = Q[state][a]
      elseif Q[state][a] == max_Q
        push!(best_action_candidates, a)
      end
    end
    # 0.1の確率でランダムに行動
    if ((0.9 < rand()) | (length(best_action_candidates) == 0)) & greedy == true
      best_action = possible_actions[rand(1:end)]
    else
    # 最も高いQ値を選択
      best_action = best_action_candidates[end] 
    end
    #@printf "-> choose action: %d \n" best_action
    push!(state_log, state)
    push!(action_log, best_action)

    end_action == best_action && return [state_log, action_log]
    # とったactionの値が次の状態の値と等価
    if D != nothing
      state, flg, i = D[state][best_action], D[state][best_action], i + 1
    else
      ns = get_random_state(num_state)
      state, flg, i = ns, ns, i + 1      
    end
    # state == goal_state && @printf("state is %d \= goal \n", state)
    # state != goal_state && @printf("state is %d \= not_goal \n", state)
  end
  return [state_log, action_log]
end

# onehot
function onehot(dim, index)
  temp = zeros(dim)
  temp[index] = 1
  return temp
end

# reinforcement learning
function eg_rl_grid()
  # ゴール状態
  goal_state = 6
  
  # 行動数 左：右：上：下：停止
  num_action = 5
  
  # 状態数
  num_state = 6
  
  # 報酬関数
  R = Dict{Any, Any}(1 => Dict(1 => -10, 2 => -10, 3 => -10, 4 => 0, 5 => -10),
                     2 => Dict(1 => -10, 2 => -10, 3 => 100, 4 => 0, 5 => -10),
                     3 => Dict(1 => 0, 2 => -10, 3 => -10, 4 => -10, 5 => -10),
                     4 => Dict(1 => 0, 2 => 0, 3 => 0, 4 => -1, 5 => -10),
                     5 => Dict(1 => -10, 2 => 0, 3 => 0, 4 => 100, 5 => -10),
                     6 => Dict(1 => -10, 2 => -10, 3 => -10, 4 => 0, 5 => 100))  
  
  # 状態遷移関数
  D =  Dict(1 => Dict( 4 => 5, 5 => 1),
  	        2 => Dict( 3 =>　6, 4 => 3, 5 => 2),
  	        3 => Dict( 1 => 4, 5 => 3),
  	        4 => Dict( 1 => 5, 2 => 3, 3 => 2, 5 => 4),
  	        5 => Dict( 2 => 4, 3 => 1, 4 => 6, 5 => 5),
  	        6 => Dict( 4 => 2, 5 => 6))

  # 重みベクトル
  w = [zeros(num_state * num_action)'']
  #w = [zeros(num_state + num_action)'']
  @show w

  # 特徴ベクトル
  f = Dict(x => Dict() for x in collect(1:num_state))
  dim = 1
  for s in collect(1:num_state)
    for a in collect(1:num_action)
      f[s][a] = onehot(num_state * num_action, dim)
      #f[s][a] = vcat(onehot(num_state, s), onehot(num_action, a))
      dim += 1
    end
  end 
  
  # 行動価値関数
  Q = Dict(x => Dict(y => 0.0 for y in collect(1:num_action)) for x in collect(1:num_state))
  
  # 割引率
  gamma = 0.8
  
  # 試行回数
  learn_count = 10000
  
  # q学習
  q_learn!(Q, R, D, w, f, gamma, learn_count, goal_state, num_state, num_action)  
  println("R function = ")
  println(R)
  println("Q function = ")
  println(Q)
  
  # 学習結果
  expert_log = []
  push!(expert_log, run_greedy(Q, R, D, num_state, num_action, 1, 6, nothing, false))
  push!(expert_log, run_greedy(Q, R, D, num_state, num_action, 2, 6, nothing, false))
  push!(expert_log, run_greedy(Q, R, D, num_state, num_action, 3, 6, nothing, false))
  push!(expert_log, run_greedy(Q, R, D, num_state, num_action, 4, 6, nothing, false))
  push!(expert_log, run_greedy(Q, R, D, num_state, num_action, 5, 6, nothing, false))
  push!(expert_log, run_greedy(Q, R, D, num_state, num_action, 6, 6, nothing, false))
  
  # 最短経路 
  println("greedy result = ")
  println(expert_log[1][1], ":", expert_log[1][2])
  println(expert_log[2][1], ":", expert_log[2][2])
  println(expert_log[3][1], ":", expert_log[3][2])
  println(expert_log[4][1], ":", expert_log[4][2])
  println(expert_log[5][1], ":", expert_log[5][2])
  println(expert_log[6][1], ":", expert_log[6][2])
  return expert_log
end

gamma = 0.8

# state
num_state = 6

# action 
num_action = 5

# エキスパートの行動ログ
expert_log = eg_rl_grid()
state_log = [x[1] for x in expert_log]
action_log =　[x[2] for x in expert_log]
  
# 特徴ベクトル
f = Dict(x => Dict() for x in collect(1:num_state))
dim = 1
for s in collect(1:num_state)
  for a in collect(1:num_action)
      f[s][a] = onehot(num_state * num_action, dim)
    dim += 1
  end
end

# u_expert
function get_u_expert(gamma, f, state_logs, action_logs)
  ue = []
  for (sl, al) in zip(state_logs, action_logs)
    ue_ = []
    for (i, (s, a)) in enumerate(zip(sl, al))
      push!(ue_, gamma^i * f[s][a])
    end
    push!(ue, sum(ue_))
  end
  return sum(ue) / length(state_logs)
end

# u 
function get_u(gamma, f, state_log, action_log)
  u = []
  for (i, (s, a)) in enumerate(zip(state_log, action_log))
    push!(u, gamma^i * f[s][a])
  end
  return sum(u) / length(u)
end

ue = get_u_expert(gamma, f, state_log, action_log)''
u = [get_u(gamma, f, state_log[3], action_log[3])'']
mu =  [u[end]]
w = [ue - mu[end]]
t = [norm(w[end])]


@show mu
@show w
@show t

# 状態遷移関数
# 状態遷移関数
D =  Dict(1 => Dict( 4 => 5, 5 => 1),
          2 => Dict( 3 =>　6, 4 => 3, 5 => 2),
         　3 => Dict( 1 => 4, 5 => 3),
          4 => Dict( 1 => 5, 2 => 3, 3 => 2, 5 => 4),
          5 => Dict( 2 => 4, 3 => 1, 4 => 6, 5 => 5),
          6 => Dict( 4 => 2, 5 => 6))

# 報酬関数の初期化
R = Dict(x => Dict() for x in collect(1:num_state))
@show R

# 行動価値関数の初期化
Q = Dict(x => Dict(y => 0.0 for y in collect(1:num_action)) for x in collect(1:num_state))

for demo_i in collect(1:100)  
  # 閾値
　 @show demo_i t[end]

  t[end] < 0.000000001 && break

  # 報酬関数の初期化
  R = Dict(x => Dict() for x in collect(1:num_state))
  # 報酬関数の更新
  update_r!(R, f, [w[end]], num_state, num_action)
  
  # 行動価値関数の初期化
　 # Q = Dict(x => Dict(y => 0.0 for y in collect(1:num_action)) for x in collect(1:num_state))
　　
  # Q学習
　　q_learn!(Q, R, D, [w[end]], f, 0.8, 1000, 6, num_state, num_action)  
  
  # best policyrun_greedy
  policy = [run_greedy(Q, R, D, num_state, num_action, ns, 6, nothing, true) for ns in collect(1:num_state)]
  
  # update
  push!(u, get_u_expert(gamma, f, [x[1] for x in policy], [x[2] for x in policy])'')
  push!(mu, mu[end] .+ ( ((u[end] - mu[end])' * (ue - mu[end])) / ((u[end] - mu[end])' * (u[end] - mu[end])) ) .* (u[end] - mu[end]))
  push!(w, ue - mu[end])
  push!(t, norm(w[end]))

end

println(" ---------- irl resutl ----------")
policy = [run_greedy(Q, R, D, num_state, num_action, ns, 6, nothing, false) for ns in collect(1:num_state)]
println(policy[1])
println(policy[2])
println(policy[3])
println(policy[4])
println(policy[5])
println(policy[6])
println("R function = ")
println(R)
println("Q function = ")
println(Q)

