# Plan: Implement Frontend Routes

## Context
The frontend currently has only 3 routes: landing/login (`/`), events list (`/events/`). The user wants all remaining routes built out so the app is functionally navigable. Pages should be unstyled/minimal but fully wired to backend APIs.

## User Decisions
- **Registration**: Self-register for events; separate enrollment per tournament
- **Draft view**: Current round displayed prominently (table, opponent, result form). Other rounds visible but read-only. Includes standings + cube link.
- **Admin**: Separate `/admin/...` pages (not inline on player pages)
- **Landing**: Simple for now — list user's events/tournaments, no smart redirects

---

## Step 1: Expand `models.ts` with all frontend types

**File**: `frontend/src/lib/models.ts`

Add interfaces matching backend entities:
```ts
Event { id, name }                              // exists
Tournament { id, name, eventId }
Registration { id, eventId, userId }
Enrollment { id, tournamentId, registrationId }
Flight { id, tournamentId }                     // inferred from schema
Draft { id, flightId, cubeId }
Player { id, draftId, enrollmentId }
Cube { id, name, cubeCobraUrl, creatorId }
Game { id, player1Id, player2Id, round, table }
Result { id, player1Wins, player2Wins, draws, gameId }
```

---

## Step 2: Update nav in `__root.tsx`

**File**: `frontend/src/routes/__root.tsx`

- Add nav links: Profile, Admin (admin-only)
- Keep existing: Home, Events

---

## Step 3: Update landing page for logged-in users

**File**: `frontend/src/routes/index.tsx`

When logged in, show:
- Welcome message
- list user's registered events — backend endpoint `GET /api/users/me/registrations`
- Link to `/events` to browse events

---

## Step 4: Create Event Detail page

**File**: `frontend/src/routes/events/$eventId.tsx`

- Fetch event via `GET /api/events/:id`
- Fetch tournaments via `GET /api/events/:id/tournaments`
- "Register for Event" button → `POST /api/events/:id/register`
- List tournaments as links to `/tournaments/$tournamentId`
- Admin: show "Create Tournament" form → `POST /api/events/:id/tournaments`

Update events list page to link event names to `/events/$eventId`.

---

## Step 5: Create Tournament Detail page

**File**: `frontend/src/routes/tournaments/$tournamentId.tsx`

- Fetch tournament via `GET /api/tournaments/:id`
- "Enroll in Tournament" button → `POST /api/tournaments/:id/enroll`
- Link back to parent event
- `GET /tournaments/:id/flights` endpoint — show list of flights for tournament
- **Gap**: No standings endpoint exposed — show placeholder

---

## Step 6: Create Draft Detail page

**File**: `frontend/src/routes/drafts/$draftId.tsx`

- Fetch draft via `GET /api/drafts/:id` (returns cubeId, flightId)
- Fetch players via `GET /api/drafts/:id/players`
- Link to cube: `/cubes/$cubeId`
- `GET /drafts/:id/games` endpoint — show sections for:
  - Current Match (with result reporting form structure)
  - All rounds display
- Show placeholder for standings table
- Deck image upload: placeholder "Coming soon"

---

## Step 7: Create Cube Detail page

**File**: `frontend/src/routes/cubes/$cubeId.tsx`

- Fetch cube via `GET /api/cubes/:id`
- Display: name, CubeCobra URL (as external link), creator info

---

## Step 8: Create Admin pages

### Admin Dashboard
**File**: `frontend/src/routes/admin/index.tsx`
- Links to manage events (list all events with edit links)
- Admin-only guard (check role, show "Access denied" for non-admins)

### Admin Event Management
**File**: `frontend/src/routes/admin/events/$eventId.tsx`
- View event details
- Create tournament form → `POST /api/events/:id/tournaments`
- Register user for event → `POST /api/events/:id/register/:userId` (with userId input)
- List tournaments with links to admin tournament pages

### Admin Tournament Management
**File**: `frontend/src/routes/admin/tournaments/$tournamentId.tsx`
- View tournament details
- Enroll user → `POST /api/tournaments/:id/enroll/:userId`
- Flights listing → `GET /api/tournaments/:id/flights`
- **Gap**: No draft management endpoints — placeholder

### Admin Draft Management
**File**: `frontend/src/routes/admin/drafts/$draftId.tsx`
- View draft details and players
- Override game results → `PUT /api/games/:id/result`
- Games listing for draft → `GET /api/drafts/:id/games`
- **Gap**: No pairings trigger endpoint — placeholder

---

## Backend Endpoint Gaps (noted, not implemented in this PR)

These endpoints are needed but don't exist yet. Frontend pages will show placeholders:

1. `GET /drafts/:id/standings` — compute and return standings for a draft
2. `GET /tournaments/:id/standings` — aggregated standings
3. Pairings trigger endpoint for admins

---

## Files to Create/Modify

### Modify:
1. `frontend/src/lib/models.ts` — add all interfaces
2. `frontend/src/routes/__root.tsx` — update nav
3. `frontend/src/routes/index.tsx` — update logged-in view
4. `frontend/src/routes/events/index.tsx` — add links to event detail pages

### Create:
5. `frontend/src/routes/events/$eventId.tsx`
6. `frontend/src/routes/tournaments/$tournamentId.tsx`
7. `frontend/src/routes/drafts/$draftId.tsx`
8. `frontend/src/routes/cubes/$cubeId.tsx`
9. `frontend/src/routes/profile.tsx`
10. `frontend/src/routes/admin/index.tsx`
11. `frontend/src/routes/admin/events/$eventId.tsx`
12. `frontend/src/routes/admin/tournaments/$tournamentId.tsx`
13. `frontend/src/routes/admin/drafts/$draftId.tsx`

---

## Verification

1. Run `cd frontend && bun run dev` — TanStack Router should auto-generate the route tree
2. Verify all routes are accessible in the browser
3. Test event detail page with a real event ID
4. Test registration and enrollment flows
5. Test admin pages with an admin user (should show admin controls)
6. Test admin pages with a player user (should show "Access denied")
7. Verify cube detail page loads with CubeCobra link
