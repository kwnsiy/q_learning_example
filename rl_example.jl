# coding: utf-8

using DataFrames


doc=
"""
Q学習: 例題
http://mnemstudio.org/path-finding-q-learning-tutorial.htm
"""

# ランダムな状態遷移
function get_random_state(R)
  return rand(1:size(R)[1])
end    

# 可能な状態遷移
function get_possible_actions(R, state)
  (state < 1) | (state > size(R)[1]) && throw("invaid state: $state")
  return find(R[state, :] .!= -1)
end

# 可能な状態遷移の中で最も高いq値を選択
function get_maxq_from_possible_actions(Q, state, possible_actions)
  return maximum([Q[state, i] for i in possible_actions])
end

# q学習
# Qは破壊的に更新
# learn_count = 試行回数
function q_learn!(Q, R, gamma, learn_count, goal_state)
  @printf "---------- qlearn ---------- \n"
  # 初期状態を定義
  state = get_random_state(R)
  
  # シミュレーター
  for i in collect(1:learn_count)
    i%100 == 0 && @printf("ite = %d \n", i)
    # 可能な状態遷移
    possible_actions = get_possible_actions(R, state)
         
    # ランダムに行動選択
    action = possible_actions[rand(1:end)]
    
    # Q値更新
    # Q(s,a) = r(s,a) + Gamma * max[Q(next_s, possible_actions)]
    next_state = action
    next_possible_actions = get_possible_actions(R, next_state)
    max_q_next_s = get_maxq_from_possible_actions(Q, next_state, next_possible_actions)
    Q[state, action] = R[state, action] + gamma * max_q_next_s
    state = next_state 

    # 終了判定
    if state == goal_state
      # 初期状態を変更して続行
      state = get_random_state(R)
    end
  end
end

# 学習結果の確認
function run_greedy(Q, R, start_state, goal_state)
  @printf "---------- run greedy ---------- \n"
  state = start_state
  state_log = []
  action_log = []
  while state != goal_state
    @printf "current state: %d \n"  state 
    # 現在の状態において可能な行動を取得
    possible_actions = get_possible_actions(R, state)
    
    # Q-value(s, a)を最大化する行動を選択
    max_Q = 0
    best_action_candidates = []
    for a in possible_actions   
      if Q[state, a] > max_Q
        push!(best_action_candidates, a)
        max_Q = Q[state, a]
      elseif Q[state, a] == max_Q
        push!(best_action_candidates, a)
      end
    end

    # 最適な候補をランダムに選択 
    best_action = best_action_candidates[rand(1:end)]
    @printf "-> choose action: %d \n" best_action
    push!(state_log, state)
    push!(action_log, best_action)
    
    # この例題では、とったactionの値が次の状態の値と等価
    state = best_action
    state == goal_state && @printf("state is %d \= goal \n", state)
    state != goal_state && @printf("state is %d \= not_goal \n", state)
  end
  return [state_log, action_log]
end

# reinforcement learning
function eg_rl_grid()
  # 報酬関数
  R = [ -1 -1 -1 -1 0 -1;
  -1 -1 -1  0 -1 100;
  -1 -1 -1  0 -1  -1;
  -1  0  0 -1  0  -1;
  0 -1 -1  0 -1 100;
  -1  0 -1 -1  0 100]

  # 行動価値関数
  Q = zeros(6, 6)
  
  # ゴール状態
  goal_state = 6

  # 割引率
  gamma = 0.8
  
  # 試行回数
  learn_count = 1000

  # q学習
  q_learn!(Q, R, gamma, learn_count, goal_state)
  
  # 学習結果
  expert_log = []
  push!(expert_log, run_greedy(Q, R, 1, 6))
  push!(expert_log, run_greedy(Q, R, 2, 6))
  push!(expert_log, run_greedy(Q, R, 3, 6))
  push!(expert_log, run_greedy(Q, R, 4, 6))
  push!(expert_log, run_greedy(Q, R, 5, 6))

  println("R function = ")
  @show DataFrame(R)

  println("Q function = ")
  @show DataFrame(Q)
  
  @show expert_log  
  return expert_log
end


# inverse reinforcement learning
function eg_irl_grid()
  # 最適経路
  expert_log = eg_rl_grid()
end

eg_rl_grid()
#eg_irl_grid()
