
 Restaurant System

A web-based restaurant management system.

⸻

 Project Overview

This project is a web-based restaurant management system developed for the Software Architectures Course.
It demonstrates a modular structure supporting menu browsing, online ordering, cart management, and checkout functionality.

⸻

 System Architecture

restaurant-system/
├─ frontend/        # HTML / CSS / JS
│   ├─ public/      # Main pages (index, restaurant, login, etc.)
│   ├─ assets/      # Styles & scripts
│   └─ ...
├─ backend/         # Flask / Node.js RESTful APIs
│   ├─ app.py       # Application entry point
│   ├─ restaurant.db # SQLite database
│   ├─ api/         # Modular API routes
│   │   ├─ menu.py  # /api/menu
│   │   ├─ cart.py  # /api/cart
│   │   └─ users.py # /api/users
│   └─ ...
└─ README.md


⸻

 System Architecture Diagram

flowchart LR
    subgraph FRONTEND [Frontend]
        A1[HTML Pages]
        A2[CSS & JS]
    end

    subgraph BACKEND [Backend]
        B1[Application Server]
        B2[REST API Endpoints]
    end

    subgraph DATABASE [Database]
        C1[(SQLite)]
    end

    A1 -->|HTTP Requests| B2
    A2 -->|Fetch API| B2
    B2 -->|SQL Queries| C1
    C1 -->|Results| B2
    B2 -->|JSON Response| A1


⸻

## My Contribution — Aysel Yarmemmedova

(Group project — individual responsibilities listed)

- Designed the visual identity (colors, typography, layout consistency)
- Implemented layouts across all website pages and menu sections
- Fixed UI inconsistencies and improved visual coherence
- Updated and optimized images and media content
- Improved usability through hierarchy and spacing adjustments
- Refactored HTML & CSS structure for maintainability
- Performed continuous UI improvements during development

⸻
## Tech Stack

- Frontend: HTML5, CSS3, JavaScript
- Backend: Flask / Node.js REST API
- Database: SQLite
- Tools: Git, VSCode, PyCharm

⸻
## Features

- Multi-page navigation
- Dynamic menu display with filters
- Shopping cart management
- Checkout and order summary
- User authentication
- Admin menu and order management

⸻









