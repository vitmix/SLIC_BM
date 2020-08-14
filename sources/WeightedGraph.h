#ifndef WEIGHTED_GRAPH_H
#define WEIGHTED_GRAPH_H

#include <unordered_map>
#include <iostream>
#include <queue>
#include "SlicAlgorithm.h"

class WeightedGraph
{
public:
    using edges = std::unordered_map<int, double>;

    WeightedGraph() = default;
    virtual ~WeightedGraph() = default;

    void construct(SlicAlgorithm::nmap & nmap)
    {
        for (auto & node : nmap) {
            for (auto & adj : node.second) {
                if (adj != node.first)
                    m_graph[node.first].insert({ adj, 0.0 });
            }
        }
    }

    double getWeight(int u, int q) {
        auto found = m_graph[u].find(q);
        return found != m_graph[u].end() ? found->second : 0.0;
    }

    edges & getWeights(int u) {
        return m_graph[u];
    }

    template <typename Func>
    void traverse(Func action)
    {
        std::queue<int> nodesToVisit;
        std::unordered_map<int, bool> visitedNodes;

        for (const auto & node : m_graph) {
            visitedNodes[node.first] = false;
        }

        int currNodeID = m_graph.begin()->first;
        nodesToVisit.push(currNodeID);

        while (!nodesToVisit.empty())
        {
            currNodeID = nodesToVisit.front();
            if (visitedNodes[currNodeID] == true) { nodesToVisit.pop(); continue; }
            visitedNodes[currNodeID] = true;
            edges & adjacents = m_graph[currNodeID];
            nodesToVisit.pop();

            for (auto & adj : adjacents)
            {
                if (!visitedNodes[adj.first]) {
                    nodesToVisit.push(adj.first);
                    adj.second = action(currNodeID, adj.first);
                    m_graph[adj.first][currNodeID] = adj.second;
                }
            }
        }
    }

    friend
    std::ostream & operator<< (
        std::ostream &        out, 
        const WeightedGraph & graph)
    {
        for (const auto & node : graph.m_graph)
        {
            std::cout << "[" << node.first << " : ";
            for (auto & adj : node.second)
                std::cout << "(n: " << adj.first << ", w: " << adj.second << ") ";
            std::cout << "]\n";
        }
        return out;
    }

private:
    std::unordered_map<int, edges> m_graph;
};

#endif // !WEIGHTED_GRAPH_H