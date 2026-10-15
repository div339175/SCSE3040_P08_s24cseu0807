# SCSE3040 — Machine Learning Operations (MLOps)
## Practical 08: REST API Deployment to Cloud Platform

**Student Name:** Divakar Maurya  
**Roll Number:** S24CSEU0807  
**Batch:** Batch 31 (EB31)  
**Date of Lab Session:** 09 October 2026  
**Institution:** Bennett University, School of Computer Science Engineering & Technology  

---

### Overview
This repository contains the production REST API for the **Delivery Time Prediction Service**, packaged with **FastAPI** and **Pydantic** input validation, serving a scikit-learn regression model trained to predict food delivery durations.

### Endpoints
- `GET /`: Service metadata, routing index, and health status link.
- `GET /health`: Health verification checking model loading state.
- `GET /docs`: Interactive Swagger UI documentation.
- `POST /predict`: Inference endpoint accepting order features with strict range validation:
  - `distance_km` (float, > 0 and <= 50)
  - `prep_time_min` (float, >= 0 and <= 120)
  - `traffic_level` (int, 1 = Low, 2 = Medium, 3 = High)
  - `rain` (int, 0 = Clear, 1 = Rain)

### Example Request
```bash
curl -X POST "https://<deployed-url>/predict" \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 5.0, "prep_time_min": 20.0, "traffic_level": 2, "rain": 0}'
```

### Example 200 OK Response
```json
{
  "predicted_delivery_min": 46.98,
  "status": "success",
  "input_data": {
    "distance_km": 5.0,
    "prep_time_min": 20.0,
    "traffic_level": 2,
    "rain": 0
  }
}
```

### Request Validation Rejection (HTTP 422)
When invalid parameters are passed (such as `traffic_level: 5` or negative distance), Pydantic automatically rejects the payload before reaching the model:
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "traffic_level"],
      "msg": "Input should be less than or equal to 3",
      "input": 5,
      "ctx": {"le": 3}
    }
  ]
}
```
