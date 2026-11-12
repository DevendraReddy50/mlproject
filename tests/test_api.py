import os
os.environ['DATABASE_URL']='sqlite://'
from fastapi.testclient import TestClient
from road_platform.api import app

def test_api_routes_and_request_validation():
    with TestClient(app) as client:
        schema=client.get('/openapi.json')
        assert schema.status_code==200
        assert '/segments/{segment_id}/predict' in schema.json()['paths']
        response=client.post('/segments',json={'id':1,'road_name':'Test','latitude':37.7,'longitude':-122.4,'pci':120})
        assert response.status_code==422
