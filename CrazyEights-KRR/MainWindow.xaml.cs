using System;
using System.IO;
using System.Collections.Generic;
using System.Net.Security;
using System.Runtime.CompilerServices;
using System.Security.Policy;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static CrazyEights_KRR.MainWindow;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Runtime.InteropServices;
using System.Windows.Media.Animation;
using System.DirectoryServices.ActiveDirectory;
using System.Net;

/*
   ██████╗██████╗  █████╗ ███████╗██╗   ██╗    ███████╗██╗ ██████╗ ██╗  ██╗████████╗███████╗
  ██╔════╝██╔══██╗██╔══██╗╚══███╔╝╚██╗ ██╔╝    ██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
  ██║     ██████╔╝███████║  ███╔╝  ╚████╔╝     █████╗  ██║██║  ███╗███████║   ██║   ███████╗
  ██║     ██╔══██╗██╔══██║ ███╔╝    ╚██╔╝      ██╔══╝  ██║██║   ██║██╔══██║   ██║   ╚════██║
  ╚██████╗██║  ██║██║  ██║███████╗   ██║       ███████╗██║╚██████╔╝██║  ██║   ██║   ███████║
   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝       ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
By ChrisPy
A Game meant to demonstrate Q-Learning by playing the card-game Crazy Eights.
*/

namespace CrazyEights_KRR
{
    public static class GlobalVar
    {
        public const int PLAYER_COUNT = 2;
        public const int GAMES = 100000;
        public const bool USE_AGENTS = true;
        public const bool TEXT_FILE_LOGS = true;
        public static StreamWriter AllLogs;
        public static StreamWriter GameLogs;
    }

    /*
        ██████  ██      
        ██   ██ ██      
        ██████  ██      
        ██   ██ ██      
        ██   ██ ███████ 
    */

    public class Agent
    {
        private Sequential model;
        private Optimizer optimizer;
        private MSELoss loss;

        private float ep; // Exploration rate epsilon
        private float gamma; // Discount factor
        private int numActions; // Number of possible actions

        public Agent()
        {
            random.manual_seed(1);
            InitializeModel(4); // Initialize with a default number of actions
            optimizer = Adam(model.parameters(), lr: 0.001);
            loss = MSELoss(reduction: Reduction.Sum);
            ep = 0.1f; // Exploration rate
            gamma = 0.99f; // Discount factor
        }

        private void InitializeModel(int numActions)
        {
            this.numActions = numActions;
            model = Sequential(
                Linear(64, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, numActions)
            );
        }

        // clones the old model and creates a new one with an updated output layer
        public void UpdateModelOutputLayer(int numActions)
        {
            if (this.numActions != numActions)
            {
                this.numActions = numActions;

                // Define the new model
                var newModel = Sequential(
                    Linear(64, 128),
                    ReLU(),
                    Linear(128, 64),
                    ReLU(),
                    Linear(64, numActions)
                );

                // Copy weights and biases from the old model to the new model
                CopyLayerParameters(model[0] as Linear, newModel[0] as Linear);
                CopyLayerParameters(model[2] as Linear, newModel[2] as Linear);

                // Replace the old model with the new model
                model = newModel;
            }
        }

        // used for copying the bias and weights from the old model to the new one
        private void CopyLayerParameters(Linear oldLayer, Linear newLayer)
        {
            newLayer.weight = oldLayer.weight.clone().AsParameter();
            newLayer.bias = oldLayer.bias.clone().AsParameter();
        }

        public int Decision(Tensor gamestate, List<int> possibleactions)
        {
            // Update the output layer to match the number of possible actions
            UpdateModelOutputLayer(possibleactions.Count);

            Random rand = new Random();

            if (rand.NextDouble() < ep)
            {
                // Exploration
                return possibleactions[rand.Next(0, possibleactions.Count)];
            }
            else
            {
                // Exploitation
                Tensor qValues = model.forward(gamestate).squeeze();
                long bestActionIndex = qValues.argmax().item<long>();
                return possibleactions[(int)bestActionIndex];
            }
        }

        public void UpdateModel(Tensor oldstate, float action, float reward, Tensor nextstate, bool complete, List<int> possibleactions)
        {
            Tensor qValues = model.forward(oldstate).squeeze();
            Tensor target = qValues.clone();

            if (!complete)
            {
                Tensor nextQValues = model.forward(nextstate).squeeze();
                float bestNextQValue = nextQValues.max().item<float>();
                action = reward + gamma * bestNextQValue;
            }
            else
            {
                action = reward;
            }

            optimizer.zero_grad();
            Tensor output = model.forward(oldstate).squeeze();
            Tensor l = loss.forward(output, target);
            l.backward();
            optimizer.step();
        }
    }

    /*
      ███████ ███    ██ ██    ██ 
      ██      ████   ██ ██    ██ 
      █████   ██ ██  ██ ██    ██ 
      ██      ██  ██ ██  ██  ██  
      ███████ ██   ████   ████   
    */

    public class Environment
    {
        private int Turn = 0;
        private int Turns = 0;
        public List<Card> playable = new List<Card>();
        public Player[] Players = new Player[GlobalVar.PLAYER_COUNT];
        private Stack<Card> Deck = new Stack<Card>();
        private Stack<Card> Pool = new Stack<Card>();
        private MainWindow M;
        private int instance = 0;
        public bool complete = false;

        // Sets up the game by generating the deck, shuffling and dealing cards
        public Environment(MainWindow M, Player[] players, int instance)
        {
            this.M = M;
            this.instance = instance;

            if (players.Length > 6) { MessageBox.Show("Too Many Players"); }

            // Loop for populating the deck
            for (int i = 0; i < 4; i++)
            {
                char suit = ' ';
                switch (i)
                {
                    case 0: suit = '♠'; break;
                    case 1: suit = '♣'; break;
                    case 2: suit = '♥'; break;
                    case 3: suit = '♦'; break;
                }
                for (int j = 0; j < 13; j++)
                {
                    char rank = ' ';
                    switch (j)
                    {
                        case 0: rank = 'A'; break;
                        case 1: rank = '2'; break;
                        case 2: rank = '3'; break;
                        case 3: rank = '4'; break;
                        case 4: rank = '5'; break;
                        case 5: rank = '6'; break;
                        case 6: rank = '7'; break;
                        case 7: rank = '8'; break;
                        case 8: rank = '9'; break;
                        case 9: rank = 'T'; break;
                        case 10: rank = 'J'; break;
                        case 11: rank = 'Q'; break;
                        case 12: rank = 'K'; break;
                    }

                    Card c = new Card(suit, rank);
                    Deck.Push(c);
                }
            }

            // Shuffle The Deck
            ShuffleDeck();

            // reset players, keep score intact
            int counter = 0;
            foreach (Player p in players)
            {
                // reset hand from previous game
                p.Hand = new List<Card>();

                // deal a new hand
                for (int j = 0; j < 5; j++)
                {
                    p.Hand.Add(Deck.Pop());
                }

                // Update with Hand
                if (p.ID == Turn) { p.UpdateHand(true); }
                else { p.UpdateHand(false); }

                // add players to this game
                Players[counter] = p;
                counter++;
            }

            // draw pool card
            Pool.Push(Deck.Pop());

            UpdateGlobalVis(false);

            // Update log
            Log("Game Setup Complete");

            // automatically starts game on initialization
            if (GlobalVar.USE_AGENTS) { AgentDecision(); }
        }

        public void EndGame()
        {
            //MessageBox.Show("Player " + Players[Turn].ID + " Wins After " + Turns + " Turns!");
            Log("#" + Turns + " Player " + Players[Turn].ID + " Wins!");

            int winningscore = 0;
            foreach(Player p in Players)
            {
                winningscore = winningscore + p.GetHandScore();
            }
            Players[Turn].updatescore(winningscore);
            Log("-Player " + Players[Turn].ID + " Scored " + winningscore + " Points!");
            GlobalVar.GameLogs.WriteLine(instance + ", " + Players[Turn].ID + "," + winningscore + "," + Turns + "," + Players[Turn].score);
            //MessageBox.Show(instance + "," + Players[Turn].ID + "," + winningscore + "," + Turns + "," + Players[Turn].score);

            UpdateGlobalVis(false);
            if (!GlobalVar.USE_AGENTS)
            {
                M.DrawCard.IsEnabled = false;
                M.PlayCard.IsEnabled = false;
            }
            complete = true;
        }

        public void PlayCard(Card card)
        {
            if (!complete)
            {
                bool choice = false;
                if (card.Rank == 'N') //eight selection
                {
                    Log("#" + Turns + " Player " + Players[Turn].ID + " NEW SUIT " + card.Suit);
                    M.SuitIndicator.Content = card.Suit;
                    NextTurn();
                }
                else if (card.Rank == '8')
                {
                    Log("#" + Turns + " Player " + Players[Turn].ID + " -> " + card.Title + " CRAZY 8!");
                    Players[Turn].Hand.Remove(card);
                    Pool.Push(card);

                    M.CurPlayerMovesVis.Items.Clear();
                    M.CurPlayerMovesVis.Items.Add("♦");
                    M.CurPlayerMovesVis.Items.Add("♥");
                    M.CurPlayerMovesVis.Items.Add("♣");
                    M.CurPlayerMovesVis.Items.Add("♠");

                    choice = true;
                    M.SuitIndicator.Content = "";
                }
                else
                {
                    Log("#" + Turns + " Player " + Players[Turn].ID + " -> " + card.Title);
                    Players[Turn].Hand.Remove(card);
                    Pool.Push(card);

                    if (Players[Turn].Hand.Count == 0)
                    {
                        EndGame();
                    }
                    else { NextTurn(); }
                    M.SuitIndicator.Content = "";
                }

                UpdateGlobalVis(choice);
            }
        }

        public void DrawCard()
        {
            // Checks if deck is empty and shuffles pool cards back into the draw deck
            if (Deck.Count == 0)
            {
                Card flip = Pool.Pop();
                Deck = Pool;
                Pool = new Stack<Card>();
                ShuffleDeck();
                Pool.Push(flip);
                Log("Pool Cards Shuffled & Returned to Draw Deck");
            }

            //draws top deck card and deals to player
            Card card = Deck.Pop();
            Log("#" + Turns + " Player " + Players[Turn].ID.ToString() + " <- " + card.Title);
            Players[Turn].Hand.Add(card);
            
            //next turn and update vis
            NextTurn();
            UpdateGlobalVis(false);
        }

        public void UpdateGlobalVis(bool choice)
        {
            if (!GlobalVar.USE_AGENTS)
            {
                playable.Clear();

                // Update Pool Card Display
                M.PoolCardVis.Content = Pool.Peek().Title;
                M.PoolAmountVis.Content = Pool.Count().ToString();

                // Update Deck Display
                M.DeckAmountVis.Content = Deck.Count().ToString();

                // Update current Player hand Selector
                for (int i = 0; i < Players.Length; i++)
                {
                    if (i == Turn) { Players[i].UpdateHand(true); }
                    else { Players[i].UpdateHand(false); }
                }

                //all Viable Plays
                if (!choice)
                {
                    M.CurPlayerMovesVis.Items.Clear();

                    if (M.SuitIndicator.Content.ToString() != "")
                    {
                        string s = M.SuitIndicator.Content.ToString();
                        foreach (Card c in Players[Turn].Hand)
                        {
                            if (c.Suit == s[0] || c.Rank == '8')
                            {
                                M.CurPlayerMovesVis.Items.Add(c.Title);
                                playable.Add(c);

                                if (!GlobalVar.USE_AGENTS)
                                {
                                    M.PlayCard.IsEnabled = true;
                                    M.DrawCard.IsEnabled = false;
                                }
                            }
                        }
                    }
                    else
                    {
                        List<Card> viable = Players[Turn].ViableCards(Pool.Peek());
                        if (viable.Count > 0)
                        {
                            if (!GlobalVar.USE_AGENTS)
                            {
                                M.PlayCard.IsEnabled = true;
                                M.DrawCard.IsEnabled = false;
                            }
                            foreach (Card x in viable)
                            {
                                M.CurPlayerMovesVis.Items.Add(x.Title);
                                playable.Add(x);
                            }
                        }
                    }

                    if (playable.Count == 0)
                    {
                        M.CurPlayerMovesVis.Items.Add("No Viable Plays");
                        M.PlayCard.IsEnabled = false;
                        M.DrawCard.IsEnabled = true;
                    }
                }
            }
        }

        public void Log(string msg)
        {
            M.LogVis.Items.Add(instance.ToString() + msg);
            if (GlobalVar.TEXT_FILE_LOGS) { GlobalVar.AllLogs.WriteLine(instance.ToString() + msg); }
        }

        public void ShuffleDeck()
        {
            // turn to list
            List<Card> cards = Deck.ToList();
            Random rng = new Random();
            int n = Deck.Count;

            //shuffle list
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                Card value = cards[k];
                cards[k] = cards[n];
                cards[n] = value;
            }
            // Clear Deck
            Deck.Clear();

            // Assign Shuffled Cards back to Deck
            foreach (Card x in cards) { Deck.Push(x); }
        }

        public void NextTurn()
        {
            if (Turn < GlobalVar.PLAYER_COUNT-1) { Turn++; }
            else { Turn = 0; }
            Turns++;
            UpdateGlobalVis(false);
            AgentDecision();
        }

        // --------------------------------------------------------------------------------------
        // Functions relevant to agent
        // --------------------------------------------------------------------------------------

        // summarises the gamestate into a tensor for the RL process
        public Tensor GameState(Player player)
        {
            List<float> gamestate = new List<float>();

            // Define the expected number of features (64)
            int expectedFeatures = 64;

            gamestate.Add(player.GetHandScore()); // the current score of the player's hand

            // Populates values for the cards in the player's hand
            for (int i = 0; i < player.Hand.Count; i++)
            {
                gamestate.Add(player.Hand[i].Values[0]);
                gamestate.Add(player.Hand[i].Values[1]);
            }

            // Populates list with pool (played) cards
            List<Card> poolcards = Pool.ToList();
            for (int i = 0; i < poolcards.Count; i++)
            {
                gamestate.Add(poolcards[i].Values[0]);
                gamestate.Add(poolcards[i].Values[1]);
            }

            // Populates the list with the number of cards in opponents' hands
            for (int i = 0; i < GlobalVar.PLAYER_COUNT - 1; i++)
            {
                gamestate.Add(Players[i].Hand.Count);
            }

            // Ensure the list has the correct number of features
            while (gamestate.Count < expectedFeatures)
            {
                gamestate.Add(0); // Pad with zeros if needed
            }

            // Truncate the list if it has more elements than expected
            if (gamestate.Count > expectedFeatures)
            {
                gamestate = gamestate.GetRange(0, expectedFeatures);
            }

            // Convert the list into an array and create the tensor to be returned
            Tensor output = tensor(gamestate.ToArray()).unsqueeze(0); // Add batch dimension

            return output;
        }

        // generates a list of unique card integers for the decision making process
        public List<int> LegalMoves(Player player)
        {
            List<int> output = new List<int>();
            List<Card> moves = player.ViableCards(Pool.Peek());

            if (moves.Count == 0)
            {
                output.Add(0);
                return output;
            }
            else
            {
                foreach (Card c in moves)
                {
                    string temp = c.Values[0].ToString() + c.Values[1].ToString();
                    output.Add(int.Parse(temp));
                }
            }
            return output;
        }

        public List<int> AgentEightChoice()
        {
            List<int> output = new List<int>();
            output.Add(1);
            output.Add(2);
            output.Add(3);
            output.Add(4);
            return output;
        }

        // takes a card being played and calculates the reward for that play given the game state
        public float CalculateReward(List<int> possibleactions, int decision, Player player)
        {
            int total = 0;

            // decision will either be a draw or an eight selection
            if (decision < 10)
            {
                //action if suit choice
                int[] suitcounts = new int[4];
                foreach (Card c in player.Hand)
                {
                    if (c.Values[1] == 0) { suitcounts[0] += 1; }
                    else if (c.Values[1] == 1) { suitcounts[1] += 1; }
                    else if (c.Values[1] == 2) { suitcounts[2] += 1; }
                    else if (c.Values[1] == 3) { suitcounts[3] += 1; }
                }

                int maxvalue = suitcounts.Max();
                int maxIndex = suitcounts.ToList().IndexOf(maxvalue);
                if (maxIndex + 1 == decision) { total += 10; }

                int minvalue = suitcounts.Min();
                int minIndex = suitcounts.ToList().IndexOf(minvalue);
                if (minIndex + 1 == decision) { total += -10; }
            }
            else
            {
                string data = decision.ToString();
                int _suit = data[data.Length - 1];
                data = data.Remove(data.Length - 1);
                int _rank = int.Parse(data);

                // finds the chosen card in the players hand
                foreach (Card c in player.Hand)
                {
                    if (Enumerable.SequenceEqual(c.Values, [_rank, _suit]))
                    {
                        if (_rank == 8) { total += 20; }
                        else if (c.Values[0] > 10) { total += 10; }
                        else { total += c.Values[0]; }
                    }

                    if (player.Hand.Count == 1 && Enumerable.SequenceEqual(player.Hand[0].Values, [_rank, _suit]))
                    {
                        return player.score * 10;
                    }
                }
            }

            // subtracts the number of turns the player has had to prevent long play
            total -= Turns / GlobalVar.PLAYER_COUNT;

            return total;
        }

        public void AgentDecision()
        {
            Player p = Players[Turn];
            List<int> moves = LegalMoves(p);
            Tensor startstate = GameState(p);
            int decision = p.Agent.Decision(startstate, moves);
            bool lastcard = false;

            // checks if the agent will win the game this turn and will delay playing the card to let the agent reach its target
            if (p.Hand.Count == 1 && decision != 0) { lastcard = true; }

            // checks if the agent has decided to play an eight
            if (decision >= 80 && decision <= 83)
            {
                char suit = ' ';
                switch (decision)
                {
                    case 80: suit = '♦'; break;
                    case 81: suit = '♥'; break;
                    case 82: suit = '♣'; break;
                    case 83: suit = '♠'; break;
                }
                foreach (Card c in p.Hand)
                {
                    if (c.Suit == suit) { PlayCard(c); break; }
                }

                List<int> eightmoves = AgentEightChoice();
                Tensor midstate = GameState(p);

                // updates agent for the suit decision
                p.Agent.UpdateModel(startstate, decision, 20, midstate, lastcard, eightmoves);

                decision = p.Agent.Decision(midstate, eightmoves);

                suit = ' ';
                switch (decision)
                {
                    case 1: suit = '♦'; break;
                    case 2: suit = '♥'; break;
                    case 3: suit = '♣'; break;
                    case 4: suit = '♠'; break;
                }
                Card change = new Card(suit, 'N');

                // play suit selection
                PlayCard(change);

                // update model
                List<int> endmoves = LegalMoves(p);
                p.Agent.UpdateModel(midstate, decision, CalculateReward(moves, decision, p), GameState(p), lastcard, endmoves);
            }
            else if (decision == 0) // The agent chooses to draw a card
            {
                DrawCard();
                p.Agent.UpdateModel(startstate, decision, 0, GameState(p), lastcard, LegalMoves(p));
            }
            else // the agent has decided to make a legal move that is not drawing or playing an 8
            {
                string data = decision.ToString();
                string _suit = data[data.Length - 1].ToString();
                data = data.Remove(data.Length - 1);
                string _rank = data;

                // finds the chosen card in the players hand
                foreach (Card c in Players[Turn].Hand)
                {
                    if (Enumerable.SequenceEqual(c.Values, [int.Parse(_rank), int.Parse(_suit)]))
                    {
                        PlayCard(c);
                        List<int> endmoves = LegalMoves(p);
                        p.Agent.UpdateModel(startstate, decision, CalculateReward(endmoves, decision, p), GameState(p), lastcard, endmoves);
                        break;
                    }
                }

            }
        }
        
    }

    public partial class MainWindow : Window
    {

        /*
           ██████  █████  ██████  ██████  
          ██      ██   ██ ██   ██ ██   ██ 
          ██      ███████ ██████  ██   ██ 
          ██      ██   ██ ██   ██ ██   ██ 
           ██████ ██   ██ ██   ██ ██████  
        */

        public class Card
        {
            public char Suit;
            public char Rank;
            public int[] Values = new int[2];
            public string Title;

            public Card(char suit, char rank)
            {
                Suit = suit;
                Rank = rank;

                // converts suit and rank into numerical values for use in the model
                if (!int.TryParse(Rank.ToString(), out int j))
                {
                    if (Rank == 'A') { Values[0] = 1; }
                    else if (Rank == 'T') { Values[0] = 10; }
                    else if (Rank == 'J') { Values[0] = 11; }
                    else if (Rank == 'Q') { Values[0] = 12; }
                    else if (Rank == 'K') { Values[0] = 13; }
                }
                else { Values[0] = j; }

                if (Suit == '♦') { Values[1] = 0; }
                else if (Suit == '♥') { Values[1] = 1; }
                else if (Suit == '♣') { Values[1] = 2; }
                else if (Suit == '♠') { Values[1] = 3; }

                Title = Rank.ToString() + Suit.ToString();
            }
        }

        /*
          ██████  ██       █████  ██    ██ ███████ ██████  
          ██   ██ ██      ██   ██  ██  ██  ██      ██   ██ 
          ██████  ██      ███████   ████   █████   ██████  
          ██      ██      ██   ██    ██    ██      ██   ██ 
          ██      ███████ ██   ██    ██    ███████ ██   ██ 
        */

        public class Player
        {
            public int ID { get; set; }

            public List<Card> Hand = new List<Card>();

            public Label Display = new Label();
            public Label Scoreboard = new Label();

            public Agent Agent = new Agent();

            public int score = 0;

            public Player(int id) { ID = id; }

            public void UpdateHand(bool turn)
            {
                string temp = "";
                foreach (Card x in Hand)
                {
                    temp = temp + x.Title + ", ";
                }
                Display.Content = temp;
                if (turn) { Display.Foreground = new SolidColorBrush(Colors.Green); }
                else { Display.Foreground = new SolidColorBrush(Colors.Black); }
            }

            public List<Card> ViableCards(Card cur)
            {
                List<Card> output = new List<Card>();
                foreach (Card x in Hand)
                {
                    if (x.Rank == cur.Rank || x.Rank == '8')
                    {
                        output.Add(x);
                    }
                    else if (x.Suit == cur.Suit)
                    {
                        output.Add(x);
                    }
                }

                return output;
            }

            public int GetHandScore()
            {
                int output = 0;

                foreach (Card c in Hand)
                {
                    if (c.Rank == 'A') { output += 1; }
                    else if (c.Rank == 'T' || c.Rank == 'J' || c.Rank == 'Q' || c.Rank == 'K') { output += 10; }
                    else if (c.Rank == '8') { output += 50; }
                    else { output += int.Parse(c.Rank.ToString()); }
                }

                return output;
            }

            public void updatescore(int s)
            {
                score = score + s;
                Scoreboard.Content = score;
            }
        }

        /*
          ███    ███  █████  ██ ███    ██ 
          ████  ████ ██   ██ ██ ████   ██ 
          ██ ████ ██ ███████ ██ ██ ██  ██ 
          ██  ██  ██ ██   ██ ██ ██  ██ ██ 
          ██      ██ ██   ██ ██ ██   ████ 
        */

        // creates an array of persistent players across games
        public Player[] players = new Player[GlobalVar.PLAYER_COUNT];

        public MainWindow()
        {
            InitializeComponent();
            if (!GlobalVar.USE_AGENTS) { PlayCard.IsEnabled = false; }
            if (GlobalVar.TEXT_FILE_LOGS)
            {
                GlobalVar.AllLogs = new StreamWriter(File.Create(DateTime.Now.ToString("yyMMddHHmmss") + "AllLogs.txt"));
                GlobalVar.GameLogs = new StreamWriter(File.Create(DateTime.Now.ToString("yyMMddHHmmss") + "GameLogs.csv"));
                GlobalVar.GameLogs.WriteLine("Instance,PlayerID,GameScore,Turns,Running");
            }

            // populates all the constant data
            for (int i = 0; i < players.Count(); i++)
            {
                Player p = new Player(i);
                // Assign Visualizers
                switch (p.ID)
                {
                    case 0: p.Display = p1Display; p.Scoreboard = p1Score; break;
                    case 1: p.Display = p2Display; p.Scoreboard = p2Score; break;
                    case 2: p.Display = p3Display; p.Scoreboard = p3Score; break;
                    case 3: p.Display = p4Display; p.Scoreboard = p4Score; break;
                    case 4: p.Display = p5Display; p.Scoreboard = p5Score; break;
                    case 5: p.Display = p6Display; p.Scoreboard = p6Score; break;
                }

                // Add Player to Players in Game
                p.Scoreboard.Content = "0";
                players[i] = p;
            }

            Loaded += MainWindow_Loaded;
        }

        private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            int numberOfGames = GlobalVar.GAMES;
            await RunGamesAsync(numberOfGames);
        }

        public async Task RunGamesAsync(int games)
        {
            for (int i = 0; i < games; i++)
            {
                Environment env = new Environment(this, players, i);

                while (!env.complete)
                {
                    await Task.Delay(100);
                }
                //MessageBox.Show("game " + i + " complete");
            }

            PlayCard.IsEnabled = false;
            DrawCard.IsEnabled = false;

            int running = 0;
            Player winner = new Player(int.MaxValue);
            foreach (Player p in players) { if (p.score > running) { winner = p; running = winner.score; } }
            MessageBox.Show("After " + GlobalVar.GAMES + ", Player " + winner.ID + " is the winner! Score: " + winner.score);
        }

        /*
          ███████ ██    ██ ███████ ███    ██ ████████ ███████ 
          ██      ██    ██ ██      ████   ██    ██    ██      
          █████   ██    ██ █████   ██ ██  ██    ██    ███████ 
          ██       ██  ██  ██      ██  ██ ██    ██         ██ 
          ███████   ████   ███████ ██   ████    ██    ███████ 
        */

        private void PlayCard_Click(object sender, RoutedEventArgs e)
        {
            if (GlobalVar.USE_AGENTS)
            {
                //env.AgentDecision();
            }/*
            else
            {

                //grabs card from playable list and passes to PlayCard
                string chosencard = CurPlayerMovesVis.SelectedItem as string;
                if (chosencard != null)
                {
                    if (chosencard == "♦" || chosencard == "♥" || chosencard == "♣" || chosencard == "♠")
                    {
                        Card c = new Card(chosencard[0], 'N');
                        env.PlayCard(c);
                    }
                    else
                    {
                        foreach (Card card in env.playable)
                        {
                            if (card.Title == chosencard) { env.PlayCard(card); break; }
                        }
                    }
                }
                else { env.Log("You Must Select a Card!"); }
            }*/
        }

        private void DrawCard_Click(object sender, RoutedEventArgs e)
        {
            //env.DrawCard();
        }
    }
}