"""Routing module for Polar Navigator AI."""
from app.routing.fuel_model import fuel_model, PolarFuelModel
from app.routing.astar import AStarRouter, haversine_nm
from app.routing.dijkstra import DijkstraRouter
from app.routing.optimizer import route_optimizer, RouteOptimizer
