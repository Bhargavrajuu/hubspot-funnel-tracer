# HubSpot Funnel Tracer

AI solution 03 of 14 · Marketing · built for **HubSpot**

Prepared by **K Bhargav Raju**.

## Problem statement

Management observes high lead generation but relatively low revenue growth. Build an AI analytics system that traces the customer journey from campaign exposure through lead generation, conversion and revenue, identifying where the marketing funnel is losing value.

## Why HubSpot

HubSpot pulls in a huge volume of inbound leads through content, free tools and ads, so turning those leads into paying customers is its core growth problem. This tool follows every lead to revenue and shows where value leaks.

## Algorithm

1. **Collect CRM touchpoints**: Ad exposure, visits, downloads, demos, sales calls, deals
2. **Stitch journeys**: Order every touch per lead into a path
3. **Measure funnel stages**: Lead → MQL → SQL → Opportunity → Customer
4. **Markov chain attribution**: Removal effect of each channel on the chance to convert
5. **Compare with last-touch**: Show what the default report gets wrong
6. **Lead scoring model**: Logistic regression on lead traits and follow-up speed
7. **Find the leaks**: Biggest stage drop and channels with leads but no revenue
8. **LLM summary**: AI analyst explains fixes

## Data

6,000 leads from one quarter with every marketing touch, sales stage and closed revenue (sample data). All figures are illustrative and are not HubSpot's actual numbers.

## How to use

Open `index.html` in any web browser. No installation is needed; the whole tool is one self-contained file.

The controls panel lets you change lead score cutoff for sales, follow-up time target, and the analysis reruns instantly. "New sample data" generates a fresh dataset.

## AI analyst panel

The "Ask the AI analyst" panel answers questions about the results in plain English through Claude. It works when the page is opened as a Claude artifact on claude.ai. Opened from GitHub, GitHub Pages or a local folder, all the models, charts and controls still run, and the panel shows that it is unavailable.
