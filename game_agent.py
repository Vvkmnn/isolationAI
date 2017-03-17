"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import isolation


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Heuristic 1: Available moves
    # Same as one in the material, i.e. #my_moves
    def open_moves(myMoves):
        """
        How many moves are there? 
        """
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        return float(len(myMoves))

    # Heuristic 2: Available moves less opponent moves
    # Borrowed from sample_players.py, yields the #my_moves - #opponent_moves
    # eval fn
    def difference_moves(myMoves, opponentMoves):
        """
        Difference between your moves and the opponent moves. 
        """
        # All legal moves are equal. We calculate the difference in the number
        # of legal moves between the player and the opponent
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        #print('You have {} moves, they have {} moves'.format(myMoves, opponentMoves))
        return float(len(myMoves) - len(opponentMoves))

    # Heuristic 3: Quadratic version of Heuristic 2
    # Similar to above, but squares both side to account for negatives.
    def quadratic_difference_moves(myMoves, opponentMoves):
        """
        Difference between your moves and opponent moves, squared. 
        """
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        return float(len(myMoves) * len(myMoves) -
                     len(opponentMoves) * len(opponentMoves))

    # Heuristic 4: Punishing the opponent!
    # As hinted at in the lecture, what happens when we weight the
    # opponent_moves more heavily?
    def weighted_difference_moves(myMoves, opponentMoves, weight):
        """
        Weighting opponent moves as more dangerous. 
        """
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        return float(len(myMoves) - (weight * len(opponentMoves)))

    # Heuristic 5: Punishing the opponent!
    # As hinted at in the lecture, what happens when we weight the
    # opponent_moves more heavily?
    def avoid_edges(myMoves, opponentMoves):
        """
        Make the center of the board less valuable. 
        """
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        # how many blanks?
        open_positions = game.get_blank_spaces()

        # how far can we go from the center of the square board?
        width = game.width / 2

        # where is he?
        oppPosition = game.get_player_location(game.get_opponent(player))

        # How far is he from the edge?
        oppHorizontalDistance = abs(width - oppPosition[0])
        oppVerticalDistance = abs(width - oppPosition[1])

        # where are you?
        myPosition = game.get_player_location(player)

        # How far are you from the edge?
        myHorizontalDistance = abs(width - myPosition[0])
        myVerticalDistance = abs(width - myPosition[1])

        # Aggregate that stuff!
        return float(len(myMoves) + myHorizontalDistance + myVerticalDistance) - (len(opponentMoves) + oppHorizontalDistance + oppVerticalDistance)

    # Final Score: Weighted average of all heuristics
    # Assign weights to all the heuristics, and calculates the weighted
    # average.
    def final_score(heur1_weight,
                    heur2_weight,
                    heur3_weight,
                    heur4_weight,
                    heur5_weight):
        """
        Weight all scores by certain values to favor certain evaluation functions.
        """
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        # check the weights!
        if heur1_weight + heur2_weight + heur3_weight + heur4_weight + heur5_weight != 1.0:
            raise ValueError(
                "the weights in the do not add up to 1; check your weights!")
        else:
            # everything's fine
            weights = [heur1_weight, heur2_weight,
                       heur3_weight, heur4_weight, heur5_weight]

        # which heuristics are you using?
        functions = [difference_moves(myMoves, opponentMoves),
                     quadratic_difference_moves(myMoves, opponentMoves),
                     open_moves(myMoves),
                     weighted_difference_moves(myMoves, opponentMoves, 10),
                     avoid_edges(myMoves, opponentMoves)]

        # get the weighted score
        return (sum([i * j for i, j in zip(weights, functions)]))

    # find opponent
    opponent = game.get_opponent(player)

    # determine the opponent's legal moves and the player's legal moves
    opponentMoves = game.get_legal_moves(opponent)
    myMoves = game.get_legal_moves(player)

    # return the final score (weights must add up to 1)
    # just picked these, tweak to see success in tournament.py!

    # for final_score(0.4, 0.3, 0.3) with quasi-equal weights for all functions:
    # ID_Improved         65.00%
    # Student             57.86%
    # Nope, let's start picking

    # for final_score(0.1,0.1,0.8) favoring the quadratic variant:
    # ID_Improved         52.14%
    # Student             54.29%
    # Okay, quadratics are good, but not by much

    # for final_score(0,0,0, 1) favoring penalizing opp moves:
    # ID_Improved         52.14%
    # Student             52.86%
    # Bad again, let's try an equal weight?

    # for final_score(0.25,0.25,0.25,0.25) favoring all heuristics:
    # ID_Improved         59.29%
    # Student             66.43%
    # Alright, weighting them all was worth it! Let's add another...

    # for final_score(0, 0, 0, 0, 1) favoring avoid_edges:
    # ID_Improved         57.86%
    # Student             47.14%
    # A new heuristic appeared! And it sucks! Maybein the average?

    # for final_score(0.2, 0.2, 0.2, 0.2, 0.2) favoring avoid_edges:
    # ID_Improved         71.43%
    # Student             47.14%
    # Nope, it's shit. Let's drop it and favor the original

    # for final_score(0.25, 0.25, 0.25, 0.25, 0) favoring everything except avoid_edges:
    # ID_Improved         62.86%
    # Student             48.57%
    # Wow, we got way worse. Maybe let's favor the eval fns we know work...

    # for final_score(0.3, 0.3, 0.25, 0.15, 0) favoring original eval fns
    # ID_Improved         54.29%
    # Student             57.14%
    # Alright, back in the black. Maybe sneak that last one in there with a
    # tiny weight?

    # for final_score(0.3, 0.3, 0.25, 0.1, 0.05) favoring original eval fns
    # ID_Improved         67.14%
    # Student             61.43%
    # Nope, it just sucks. Fuck it, drop it.

    return final_score(0.3, 0.3, 0.25, 0.15, 0.0)


class CustomPlayer:
    """Game-playing agent that chooses a move using a given evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. The goal is to finish
    and test this player to make sure it returns a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        plys in the game tree to explore for search. 

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    # initialize the agent with a depth 3, the custom_score eval fn, iterative search
    # via the minimax method, and a 10 second timeout
    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
            forfeit the game due to timeout. You must return _before_ the
            timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        # How much time do we have?
        self.time_left = time_left

        # Perform any required initializations (i.e., an opening book)
        # Opening book
        # TODO: open_book = []? # best starting moves?

        # Setup
        # Lets make a list to recieve all potential moves suggested from the
        # methods
        potential_moves = list()

        # Start
        # Returning immediately if there are no legal moves
        if len(legal_moves) == 0:
            return (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # Search Depth
            # If depth is negative, make it some arbitrarily large number and
            # go until Timeout
            if self.search_depth < 0:
                self.search_depth = 999

            # If iterative, pick a method and go to the given depth using that
            # method
            if self.iterative:
                if self.method == 'minimax':
                    # if minimax, append potential_moves with our best moves
                    for d in range(1, self.search_depth + 1):
                        potential_moves.append(self.minimax(game, d))
                elif self.method == 'alphabeta':
                    # if alphabeta, same idea
                    for d in range(1, self.search_depth + 1):
                        potential_moves.append(self.alphabeta(game, d))

            # if it's not iterative, just grab the closest move and go
            else:
                if self.method == 'minimax':
                    # minimax for the current ply
                    potential_moves.append(self.minimax(game,
                                                        self.search_depth))
                elif self.method == 'alphabeta':
                    # alphabeta for the current ply
                    potential_moves.append(self.alphabeta(game,
                                                          self.search_depth))

        # if we're out of time, raise the Timeout Exception
        except Timeout:
            pass

        # if we still have time, and potential_moves, get the best move we have
        if potential_moves != []:
            best_move = max(potential_moves)
            return best_move[1]
        else:
            # lol fail
            return (-1, -1)

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        # Check if we're out of time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Setup
        scores = []
        final_move = (-1, -1)

        # Which player?
        if maximizing_player:
            # if it's the active player, he wants the maximum score (whichever
            # player)
            player = game.active_player
            final_score = float("-inf")
        else:
            # if it's the inactive player, he wants the minimum score
            # (whichever player)
            player = game.inactive_player
            final_score = float("+inf")

        # Let's get all of his moves
        legal_moves = game.get_legal_moves()

        # if no moves, deafault to that (-1,-1)
        if len(legal_moves) == 0:
            score, selectedMove = self.score(game, player), (-1, -1)

        # if only 1 move deep, forecast just the next game for all legal moves
        if depth == 1:
            for move in legal_moves:
                # forecast the move! neat function.
                nextGame = game.forecast_move(move)
                # check the score of that game, using that players score fn
                score = self.score(nextGame, player)
                # append that score to the list we made earlier
                scores.append(score)

                # if that score is better that +/- inf (for max or min player)
                # use that!
                if (maximizing_player and score > final_score) or (not maximizing_player and score < final_score):
                    final_score, final_move = score, move

        # if we can go deeper
        elif depth > 1:

            # for every legal move
            for move in legal_moves:
                # forecast the next game for a given
                nextGame = game.forecast_move(move)

                # call minimax on itself, get the score of the next game to the final score
                # because of depth - 1, will iterate until we hit depth = 1
                # then return the last thing in that new list (score of the
                # latest game)
                score = self.minimax(nextGame, depth - 1,
                                     not maximizing_player)[0]
                scores.append(score)

                # if that score is better that +/- inf (for max or min player)
                # use that!
                if (maximizing_player and score > final_score) or (not maximizing_player and score < final_score):
                    final_score, final_move = score, move

        # return the result
        #print('\nCurrent player is maximizing? {}'.format(maximizing_player))
        #print('Score is {} from {}, at depth {}'.format(score, scores, depth))
        return final_score, final_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """

        # Check if we're out of time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Setup
        scores = []
        explore = True
        final_move = (-1, -1)

        # Use the minimax logic from before to set ranges for maxing and
        # minimizing respectively
        if maximizing_player:
            player = game.active_player
            final_score = float("-inf")
        else:
            player = game.inactive_player
            final_score = float("+inf")

        # return score and (-1, -1) if we're out of moves so we can lose
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (self.score(game, player), (-1, -1))

        # for depth = 1
        if depth == 1:
            for move in legal_moves:
                if explore:
                    # check the next game in the tree
                    nextGame = game.forecast_move(move)
                    # find that games score
                    score = self.score(nextGame, player)
                    # add that to the score list
                    scores.append(score)

                    # if we're minimizing update the beta, and get us that
                    # score and move
                    if not maximizing_player:
                        # get the smallest score; that's the beta, prune the
                        # rest
                        beta = min(beta, score)
                        if score < final_score:
                            final_move = move
                            final_score = score
                    # if we're maximizing update alpha, and get us that score
                    # and move
                    else:
                        # get the largest score; that's the alpha, prune the
                        # rest
                        alpha = max(alpha, score)
                        if score > final_score:
                            final_move = move
                            final_score = score

                    # if alpha is greater than beta, stop searching; we're lost
                    if alpha >= beta:
                        explore = False

            return final_score, final_move

        # now for all the depths!
        elif depth > 1:
            for move in legal_moves:
                if explore:
                    nextGame = game.forecast_move(move)
                    # let's use that recursive trick to call ourselves on the
                    # next depth over and over until depth = 1
                    score = self.alphabeta(nextGame, depth - 1,
                                           alpha,
                                           beta, not maximizing_player)[0]
                    scores.append(score)

                    # do exactly what we did above, but now with recursion!
                    # (Since this calls for any depth > 1)
                    if not maximizing_player:
                        beta = min(beta, score)
                        if score < final_score:
                            final_move = move
                            final_score = score

                    else:
                        alpha = max(alpha, score)
                        if score > final_score:
                            final_move = move
                            final_score = score

                    # interrupt again if alpha >= beta
                    if alpha >= beta:
                        explore = False

            # return results!
            return final_score, final_move
