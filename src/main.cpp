#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <random>
#include <algorithm>
#include "arial.h"

#define MAX_NB_PLAYERS 1001
#define MAX_NB_INPUTS 22
#define MAX_NB_HIDDEN_LAYERS 10
#define MAX_NB_PER_HIDDEN 50
#define MAX_NB_OUTPUTS 5

#define MAX_NB_TICKS 1000
#define MAX_GENS 1000

#define PD(x) std::cout << x << std::endl;

const int MaxWeightArraySize = (MAX_NB_INPUTS * MAX_NB_PER_HIDDEN) +
                               MAX_NB_PER_HIDDEN * MAX_NB_PER_HIDDEN * (MAX_NB_HIDDEN_LAYERS - 1) +
                               MAX_NB_PER_HIDDEN * MAX_NB_OUTPUTS;
const int MaxBiasesArraySize = MAX_NB_HIDDEN_LAYERS * MAX_NB_PER_HIDDEN + MAX_NB_OUTPUTS;
const double activation_threshold = 3;

sf::Font font;

class Coord
{
public:
    double x, y;
    Coord(double x_inp, double y_inp)
    {
        x = x_inp, y = y_inp;
    }
};

class Player
{
public:
    double x, y, x_speed, y_speed;
    Coord get_player_coord()
    {
        return Coord(x, y);
    }
};

static std::random_device rd;
static std::mt19937 generator(rd());
static std::uniform_real_distribution<double> distrib_d(0.0, 1.0);
static std::uniform_int_distribution<int> distrib_i(0, 1);

double get_rand_double01()
{
    return distrib_d(generator);
}

int get_rand_int01()
{
    return distrib_i(generator);
}

class NeuralNetwork
{
public:
    NeuralNetwork(int inp, int per_hid, int out, int nb_hid, bool randomize = true)
    {
        if (inp > MAX_NB_INPUTS || per_hid > MAX_NB_PER_HIDDEN ||
            out > MAX_NB_OUTPUTS || nb_hid > MAX_NB_HIDDEN_LAYERS ||
            inp < 0 || per_hid < 0 || out < 0 || nb_hid < 0)
        {
            throw std::invalid_argument("Invalid or out of range dimensions");
        }
        nb_inputs = inp, nb_per_hidden = per_hid, nb_outputs = out, nb_hidden_layers = nb_hid;
        nb_of_weigths = (nb_inputs * nb_per_hidden) +
                        nb_per_hidden * nb_per_hidden * (nb_hidden_layers - 1) +
                        nb_per_hidden * nb_hidden_layers;
        nb_of_biases = nb_hidden_layers * nb_per_hidden + nb_outputs;
        if (randomize)
        {
            init_randomize();
        }
    }

    void init_randomize()
    {
        for (int i = 0; i < nb_of_weigths + 1; i++)
        {
            weights.push_back(4 * get_rand_double01() - 2);
        }
        for (int i = 0; i < nb_of_biases + 1; i++)
        {
            biases.push_back(4 * get_rand_double01() - 2);
        }
    }

    std::vector<double> run_network(std::vector<double> inputs)
    {
        // input part
        std::array<double, MAX_NB_PER_HIDDEN> next_values;
        for (int i = 0; i < nb_per_hidden; i++) // next_values reset
        {
            next_values[i] = 0;
        }
        for (int i_inp = 0; i_inp < nb_inputs; i_inp++) // weight calculations
        {
            for (int i_hidden = 0; i_hidden < nb_per_hidden; i_hidden++)
            {
                next_values[i_hidden] += inputs[i_inp] * weights[get_w(0, i_inp, i_hidden)];
            }
        }
        for (int i_hidden = 0; i_hidden < nb_per_hidden; i_hidden++) // bias calculations
        {
            next_values[i_hidden] += biases[get_b(1, i_hidden)];
        }

        // hidden_layers part
        std::array<double, MAX_NB_PER_HIDDEN> current_values;
        for (int i_s_layer = 1; i_s_layer < nb_hidden_layers; i_s_layer++)
        {
            for (int i = 0; i < nb_per_hidden; i++) // previous values become current values
            {
                current_values[i] = next_values[i];
                next_values[i] = 0;
            }
            for (int i_s_hidden = 0; i_s_hidden < nb_per_hidden; i_s_hidden++) // weight calculations
            {
                for (int i_e_hidden = 0; i_e_hidden < nb_per_hidden; i_e_hidden++)
                {
                    next_values[i_e_hidden] += current_values[i_s_hidden] * weights[get_w(i_s_layer, i_s_hidden, i_e_hidden)];
                }
            }
            for (int i_e_hidden = 0; i_e_hidden < nb_per_hidden; i_e_hidden++) // bias calculations
            {
                next_values[i_e_hidden] += biases[get_b(i_s_layer + 1, i_e_hidden)];
                next_values[i_e_hidden] = reLu(next_values[i_e_hidden]);
            }
        }

        // output part
        std::vector<double> output_values;
        for (int i = 0; i < nb_per_hidden; i++) // previous values become current values
        {
            current_values[i] = next_values[i];
        }
        for (int i = 0; i < nb_outputs; i++)
        {
            output_values.push_back(0);
        }
        for (int i_s_hidden = 0; i_s_hidden < nb_per_hidden; i_s_hidden++) // weight calculations
        {
            for (int i_output = 0; i_output < nb_outputs; i_output++)
            {
                output_values[i_output] += current_values[i_s_hidden] * weights[get_w(nb_hidden_layers, i_s_hidden, i_output)];
            }
        }
        for (int i_output = 0; i_output < nb_outputs; i_output++) // bias calculations
        {
            output_values[i_output] += biases[get_b(nb_hidden_layers + 1, i_output)];
            output_values[i_output] = reLu(output_values[i_output]);
        }

        return output_values;
    }

    NeuralNetwork cross_over_with(NeuralNetwork other)
    {
        NeuralNetwork resultingNN(nb_inputs, nb_per_hidden, nb_outputs, nb_hidden_layers, false);
        for (int i_weight = 0; i_weight < nb_of_weigths; i_weight++)
        {
            if (get_rand_int01() == 0)
            {
                resultingNN.weights.push_back(weights[i_weight]);
            }
            else
            {
                resultingNN.weights.push_back(other.weights[i_weight]);
            }
        }
        for (int i_bias = 0; i_bias < nb_of_biases; i_bias++)
        {
            if (get_rand_int01() == 0)
            {
                resultingNN.biases.push_back(biases[i_bias]);
            }
            else
            {
                resultingNN.biases.push_back(other.biases[i_bias]);
            }
        }
        return resultingNN;
    }

    void mutate(double prob = 0.001)
    {
        for (int i_weight = 0; i_weight < nb_of_weigths; i_weight++)
        {
            if (get_rand_double01() < prob)
            {
                weights[i_weight] += get_rand_double01() * 4 - 2;
            }
        }
        for (int i_bias = 0; i_bias < nb_of_biases; i_bias++)
        {
            if (get_rand_double01() < prob)
            {
                biases[i_bias] += get_rand_double01() * 4 - 2;
            }
        }
    }

    int get_nb_outputs()
    {
        return nb_outputs;
    }

private:
    int nb_inputs, nb_per_hidden, nb_outputs, nb_hidden_layers;
    int nb_of_weigths, nb_of_biases;
    std::vector<double> weights;
    std::vector<double> biases;

    int get_idx_w(int start_layer, int start_id, int end_id)
    {
        if (start_layer == 0)
        {
            return start_id * nb_per_hidden + end_id;
        }
        else if (start_layer == nb_hidden_layers)
        {
            return nb_inputs * nb_per_hidden +
                   (nb_hidden_layers - 1) * nb_per_hidden * nb_hidden_layers +
                   start_id * nb_outputs + end_id;
        }
        else
        {
            return nb_inputs * nb_per_hidden +
                   (start_layer - 1) * nb_per_hidden * nb_hidden_layers +
                   start_id * nb_per_hidden + end_id;
        }
    }

    int get_idx_b(int layer, int neuron_id)
    {
        if (layer == nb_hidden_layers + 1)
        {
            return nb_hidden_layers * nb_per_hidden + neuron_id;
        }
        else
        {
            return (layer - 1) * nb_per_hidden + neuron_id;
        }
    }

    double get_w(int start_layer, int start_id, int end_id)
    {
        /*
        std::cout << "w : " << get_idx_w(start_layer, start_id, end_id)
                  << " (" << start_layer << " " << start_id << " " << end_id << ")"
                  << std::endl;
        */
        return weights[get_idx_w(start_layer, start_id, end_id)];
    }

    double get_b(int layer, int neuron_id)
    {
        /*
        std::cout << "b : " << get_idx_b(layer, neuron_id)
                  << " (" << layer << " " << neuron_id << ")"
                  << std::endl;
        ;
        */
        return biases[get_idx_b(layer, neuron_id)];
    }

    double reLu(double x)
    {
        return (x > 0 ? x : 0);
    }

    double sigmoid(double x)
    {
        return (double)1 / (1 + exp(-x));
    }
};

std::vector<NeuralNetwork> wheel_selection(std::vector<NeuralNetwork> networks, int nbAIs, std::array<double, MAX_NB_PLAYERS> scores)
{
    std::vector<NeuralNetwork> next_gen;

    double score_sum = 0;
    for (int i_net = 0; i_net < nbAIs; i_net++)
    {
        score_sum += scores[i_net];
    }

    for (int i_next_net = 0; i_next_net < nbAIs; i_next_net++)
    {
        double stop_score = get_rand_double01() * score_sum;
        double curr_score_sum = 0;
        int f_i_net = 0;
        while (curr_score_sum < stop_score)
        {
            curr_score_sum += scores[f_i_net];
            f_i_net++;
        }
        f_i_net--;

        stop_score = get_rand_double01() * score_sum;
        curr_score_sum = 0;
        int s_i_net = 0;
        while (curr_score_sum < stop_score)
        {
            curr_score_sum += scores[s_i_net];
            s_i_net++;
        }
        s_i_net--;
        NeuralNetwork built_net = networks[f_i_net].cross_over_with(networks[s_i_net]);
        built_net.mutate();
        next_gen.push_back(built_net);
    }
    return next_gen;
}

std::vector<NeuralNetwork> rank_wheel_selection(std::vector<NeuralNetwork> networks, int nbAIs, std::array<double, MAX_NB_PLAYERS> rankings)
{
    std::vector<NeuralNetwork> next_gen;

    int rankings_sum = nbAIs * (nbAIs + 1) / 2;

    for (int i_next_net = 0; i_next_net < nbAIs; i_next_net++)
    {
        double stop_score = get_rand_double01() * rankings_sum;
        double curr_score_sum = 0;
        int f_rank = 0;
        while (curr_score_sum <= stop_score)
        {
            curr_score_sum += nbAIs - f_rank;
            f_rank++;
        }
        f_rank--;

        stop_score = get_rand_double01() * rankings_sum;
        curr_score_sum = 0;
        int s_rank = 0;
        while (curr_score_sum <= stop_score)
        {
            curr_score_sum += nbAIs - s_rank;
            s_rank++;
        }
        s_rank--;
        NeuralNetwork built_net = networks[rankings[f_rank]].cross_over_with(networks[rankings[s_rank]]);
        built_net.mutate();
        next_gen.push_back(built_net);
    }
    return next_gen;
}

class SlidingGameReport
{
public:
    int nb_players;
    std::array<double, MAX_NB_PLAYERS> scores;
    std::array<std::vector<Coord>, MAX_NB_PLAYERS> recording;
    int nb_ticks_recorded;
    std::array<double, MAX_NB_PLAYERS> rankings;
    void make_rankings()
    {
        std::array<double, MAX_NB_PLAYERS> cur_scores;
        std::copy(scores.begin(), scores.begin() + nb_players, cur_scores.begin());
        for (int i = 0; i < nb_players; i++)
        {
            rankings[i] = i;
        }
        std::sort(rankings.begin(), rankings.begin() + nb_players, [&cur_scores](int a, int b)
                  { return cur_scores[a] > cur_scores[b]; });
    }
};

class SlidingGame
{
public:
    SlidingGame(double slipperyness_arg, double accel_arg, int arena_size_arg)
    {
        slipperyness = slipperyness_arg, accel = accel_arg, arena_size = arena_size_arg;
    }

    void update_player(int i_player, std::array<bool, 5> button_presse)
    {
        players[i_player].x_speed += (button_presse[0] ? accel : 0) - (button_presse[1] ? accel : 0);
        players[i_player].y_speed += (button_presse[2] ? accel : 0) - (button_presse[3] ? accel : 0);
        players[i_player].x_speed *= slipperyness;
        players[i_player].y_speed *= slipperyness;
        players[i_player].x += players[i_player].x_speed;
        players[i_player].y += players[i_player].y_speed;
        if (is_dead(i_player))
        {
            alive[i_player] = false;
        }
    }

    void reset_player(int i_player)
    {
        players[i_player].x = 0;
        players[i_player].y = 0;
        players[i_player].x_speed = 0;
        players[i_player].y_speed = 0;
        alive[i_player] = true;
    }

    bool is_dead(int i_player)
    {
        if (players[i_player].x < -arena_size / 2 || players[i_player].x > arena_size / 2 ||
            players[i_player].y < -arena_size / 2 || players[i_player].y > arena_size / 2)
        {
            return true;
        }
        return false;
    }

    std::vector<double> get_player_data(int i_player)
    {
        std::vector<double> result;
        result.push_back(players[i_player].x);
        result.push_back(players[i_player].y);
        return result;
    }

    double get_player_score(int i_player)
    {
        double x_speed = players[i_player].x_speed, y_speed = players[i_player].y_speed;
        return sqrt(x_speed * x_speed + y_speed * y_speed);
    }

    SlidingGameReport play_AI_recorded_game(int nbAIs, std::vector<NeuralNetwork> networks)
    {
        for (int i_player = 0; i_player < nbAIs; i_player++)
        {
            reset_player(i_player);
        }
        SlidingGameReport report;
        report.nb_players = nbAIs;
        report.nb_ticks_recorded = 0;
        for (int i_player = 0; i_player < nbAIs; i_player++)
        {
            report.scores[i_player] = 0;
        }

        int i_frame = 0;
        while (i_frame < MAX_NB_TICKS)
        {
            for (int i_player = 0; i_player < nbAIs; i_player++)
            {
                if (alive[i_player])
                {
                    std::vector<double> network_inps = get_player_data(i_player);

                    update_player(i_player, translate_network_output(networks[i_player].run_network(network_inps)));
                    report.scores[i_player] += get_player_score(i_player);
                    report.recording[i_player].push_back(players[i_player].get_player_coord());
                }
            }
            i_frame++;
            report.nb_ticks_recorded += 1;
        }
        return report;
    }

    std::array<bool, 5> translate_network_output(std::vector<double> net_outputs)
    {
        std::array<bool, 5> result;
        for (int i_out = 0; i_out < net_outputs.size() && i_out < 5; i_out++)
        {
            result[i_out] = net_outputs[i_out] > activation_threshold;
        }
        return result;
    }
    Player get_player(int player_id)
    {
        return players[player_id];
    }
    int get_arena_size()
    {
        return arena_size;
    }

private:
    double slipperyness;
    double accel;
    int arena_size;
    Player players[MAX_NB_PLAYERS + 1];
    bool alive[MAX_NB_PLAYERS + 1];
};

std::vector<SlidingGameReport> play_AI_sim(int nbAIs, std::vector<NeuralNetwork> starting_networks, double score_goal = INFINITY, int nb_gens = MAX_GENS)
{
    std::vector<NeuralNetwork> networks = starting_networks;
    std::vector<SlidingGameReport> recordings;

    int i_gen = 0;
    double curr_best_score = -INFINITY;
    while (i_gen < nb_gens && curr_best_score < score_goal)
    {
        SlidingGame game(0.95, 1.0, 1000);
        SlidingGameReport game_report = game.play_AI_recorded_game(nbAIs, networks);
        game_report.make_rankings();
        std::cout << "Gen  " << i_gen << " : Player " << game_report.rankings[0] << " won with score " << game_report.scores[game_report.rankings[0]] << std::endl;
        recordings.push_back(game_report);
        networks = wheel_selection(networks, nbAIs, game_report.scores);
        i_gen++;
    }
    return recordings;
}

int screen_arena_size, x_arena, y_arena;

sf::CircleShape draw_player(int x, int y, int game_arena_size, sf::Color col)
{
    double scale = (double)screen_arena_size / game_arena_size;
    int real_x = x * scale + x_arena;
    int real_y = y * scale + y_arena;
    sf::CircleShape dot(screen_arena_size / 100.0);
    dot.setPosition(real_x, real_y);
    dot.setFillColor(col);
    return dot;
}

sf::Text draw_label(int x, int y, int game_arena_size, std::string label_text, sf::Color col = sf::Color(0, 0, 0))
{
    double scale = (double)screen_arena_size / game_arena_size;
    int real_x = x * scale + x_arena;
    int real_y = y * scale + y_arena;

    sf::Text text;
    text.setFont(font);
    text.setString(label_text);
    text.setCharacterSize((screen_arena_size / 100.0) * 1.5);
    text.setPosition(real_x + 2, real_y + 2);
    text.setFillColor(col);
    return text;
}
std::array<bool, 5> get_rand_input(int i_network)
{
    std::array<bool, 5> input;
    static std::default_random_engine generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, 1);
    for (int i = 0; i < 5; i++)
    {
        input[i] = (bool)distribution(generator);
    }
    return input;
}

int main()
{
    int window_width = 1000, window_height = 1000;
    int mode = 0;
    int nb_AIs = 100;
    x_arena = 500, y_arena = 500, screen_arena_size = 900;

    std::vector<NeuralNetwork> networks;
    for (int i = 0; i < nb_AIs; i++)
    {
        networks.push_back(NeuralNetwork(2, 3, 5, 1));
    }

    std::vector<SlidingGameReport> game_reports = play_AI_sim(nb_AIs, networks);
    int i_frame_sim = 0;

    SlidingGame game(0.95, 1.0, 1000);

    for (int i_player = 0; i_player < nb_AIs + 1; i_player++)
    {
        game.reset_player(i_player);
    }

    if (!font.loadFromMemory(arial_ttf, arial_ttf_len))
    {
        std::cerr << "Text error" << std::endl;
        return -1;
    }
    bool show_labels = false;

    sf::RenderWindow window(
        sf::VideoMode(window_width, window_height),
        "Sliding-AI");

    window.setFramerateLimit(60);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
            if (event.type == sf::Event::Resized)
            {
                window_width = event.size.width;
                window_height = event.size.height;

                sf::FloatRect visibleArea(0, 0, window_width, window_height);
                window.setView(sf::View(visibleArea));

                screen_arena_size = std::min(window_width - 100, window_height - 100);
                x_arena = window_width / 2, y_arena = window_height / 2;
            }
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::L)
            {
                show_labels = !show_labels;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1))
        {
            mode = 0;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num2))
        {
            mode = 1;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num3))
        {
            mode = 2;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num4))
        {
            mode = 3;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num5))
        {
            mode = 4;
        }

        if (mode == 0 || mode == 2 || mode == 4)
        {
            std::array<bool, 5> inps = {
                sf::Keyboard::isKeyPressed(sf::Keyboard::Right) || sf::Keyboard::isKeyPressed(sf::Keyboard::D),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Left) || sf::Keyboard::isKeyPressed(sf::Keyboard::Q),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Down) || sf::Keyboard::isKeyPressed(sf::Keyboard::S),
                sf::Keyboard::isKeyPressed(sf::Keyboard::Up) || sf::Keyboard::isKeyPressed(sf::Keyboard::Z),
                false};
            game.update_player(0, inps);
        }
        if (mode == 1 || mode == 2)
        {
            for (int i_player = 1; i_player < nb_AIs + 1; i_player++)
            {
                std::array<bool, 5> inps = game.translate_network_output(networks[i_player - 1].run_network(game.get_player_data(i_player - 1)));
                game.update_player(i_player, inps);
            }
        }

        window.clear(sf::Color::White);

        sf::RectangleShape arena(sf::Vector2f(screen_arena_size, screen_arena_size));
        arena.setPosition(x_arena - screen_arena_size / 2, y_arena - screen_arena_size / 2);
        arena.setFillColor(sf::Color(171, 203, 230));
        window.draw(arena);

        if (mode == 0 || mode == 2 || mode == 4)
        {
            Player p_drawn = game.get_player(0);
            window.draw(draw_player(p_drawn.x, p_drawn.y, game.get_arena_size(), sf::Color(128, 0, 128)));
        }
        if (mode == 1 || mode == 2)
        {
            for (int i_player = 1; i_player < nb_AIs + 1; i_player++)
            {
                Coord coord_drawn = game.get_player(i_player).get_player_coord();
                window.draw(draw_player(coord_drawn.x, coord_drawn.y, game.get_arena_size(), sf::Color(255, 0, 0)));
            }
        }
        if (mode == 3 || mode == 4)
        {
            for (int i_IA = 0; i_IA < nb_AIs; i_IA++)
            {
                if (game_reports[game_reports.size() - 1].recording[i_IA].size() > i_frame_sim)
                {
                    Coord coord_drawn = game_reports[game_reports.size() - 1].recording[i_IA][i_frame_sim];
                    sf::Color col = (game_reports[game_reports.size() - 1].rankings[0] == i_IA ? sf::Color(255, 127, 0) : sf::Color(255, 0, 0));
                    window.draw(draw_player(coord_drawn.x, coord_drawn.y, game.get_arena_size(), col));
                    if (show_labels)
                    {
                        window.draw(draw_label(coord_drawn.x, coord_drawn.y, game.get_arena_size(), std::to_string(i_IA)));
                    }
                }
            }
            i_frame_sim = (i_frame_sim + 1) % MAX_NB_TICKS;
        }

        window.display();
    }

    return 0;
}
