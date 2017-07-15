# coding: utf-8

using DataFrames

doc=
"""
Q学習: 例題
http://mnemstudio.org/path-finding-q-learning-tutorial.htm
"""

# ランダムな状態遷移
function get_random_state(goal_state)
  return rand(1:goal_state)
end    

# 可能な状態遷移
function get_possible_actions(D, state, goal_state)
  (state < 1) | (state > goal_state) && throw("invaid state: $state")
  return D[state]
end

# 可能な状態遷移の中で最も高いq値を選択
function get_maxq_from_possible_actions(Q, state, possible_actions)
  return maximum([Q[state, i] for i in possible_actions])
end

# q学習
# Qは破壊的に更新
# learn_count = 試行回数
function q_learn!(Q, R, D, gamma, learn_count, goal_state)
  # @printf "---------- qlearn ---------- \n"
  # 初期状態を定義
  state = get_random_state(goal_state)
  
  # シミュレーター
  for i in collect(1:learn_count)
    # i%100 == 0 && @printf("ite = %d \n", i)
    # 可能な状態遷移
    possible_actions = get_possible_actions(D, state, goal_state)
         
    # ランダムに行動選択
    action = possible_actions[rand(1:end)]
    
    # Q値更新
    # Q(s,a) = r(s,a) + Gamma * max[Q(next_s, possible_actions)]
    next_state = action
    next_possible_actions = get_possible_actions(D, next_state, goal_state)
    max_q_next_s = get_maxq_from_possible_actions(Q, next_state, next_possible_actions)
    Q[state, action] = R[state, action] + gamma * max_q_next_s
    #Q[state, action] = Q[state, action] + 0.01 * (R[state, action] + gamma * max_q_next_s - Q[state, action])
    state = next_state 

    # 終了判定
    if state == goal_state
      # 初期状態を変更して続行
      state = get_random_state(goal_state)
    end
  end
end

# 学習結果の確認
function run_greedy(Q, R, D, start_state, goal_state, greedy)
  #@printf "---------- run greedy ---------- \n"
  state_log = []
  action_log = []
  flg, state = 0, start_state
  i = 0
  while flg != goal_state
    #@printf "current state: %d \n"  state - 1
    # 現在の状態において可能な行動を取得
    possible_actions = get_possible_actions(D, state, goal_state)    
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
    # 0.1の確率でランダムに行動
    if ((0.9 < rand()) | (length(best_action_candidates) == 0)) & greedy == true
      best_action = possible_actions[rand(1:end)]
    else
    # 最も高いQ値を選択
      best_action = best_action_candidates[end] 
    end
    #@printf "-> choose action: %d \n" best_action - 1
    push!(state_log, state)
    push!(action_log, best_action)
    
    # とったactionの値が次の状態の値と等価
    state, flg, i = best_action, best_action, i + 1
    #state == goal_state && @printf("state is %d \= goal \n", state)
    #state != goal_state && @printf("state is %d \= not_goal \n", state)
    i > 1000 && break
  end
  return [state_log, action_log]
end

# reinforcement learning
function eg_rl_grid()
  # ゴール状態
  goal_state = 6

  # 報酬関数
  R = [ -1 -1 -1 -1 0 -1;
  -1 -1 -1  0 -1 100;
  -1 -1 -1  0 -1  -1;
  -1  0  0 -1  0  -1;
  0 -1 -1  0 -1 100;
  -1  0 -1 -1  0 100]
  
  # 行動価値関数
  Q = zeros(6, 6)
  
  # 状態遷移関数
  D =  Dict(s => find(R[s, :] .!= -1) for s in collect(1:6))
  
  # 割引率
  gamma = 0.8
  
  # 試行回数
  learn_count = 1000

  # q学習
  q_learn!(Q, R, D, gamma, learn_count, goal_state)
  
  Q = Q / maximum(Q) * 100

  # 学習結果
  expert_log = []
  push!(expert_log, run_greedy(Q, R, D, 1, 6, false))
  push!(expert_log, run_greedy(Q, R, D, 2, 6, false))
  push!(expert_log, run_greedy(Q, R, D, 3, 6, false))
  push!(expert_log, run_greedy(Q, R, D, 4, 6, false))
  push!(expert_log, run_greedy(Q, R, D, 5, 6, false))
  push!(expert_log, run_greedy(Q, R, D, 6, 6, false))

  println("R function = ")
  @show DataFrame(R)

  println("Q function = ")
  @show DataFrame(Q)
  
  @show expert_log  
  return expert_log
end

# eg_rl_grid()
# eg_irl_grid()

# eg_rl_grid()

