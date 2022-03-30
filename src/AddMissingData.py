# detect bluetooth etc changes and add dem in

# detect changes in proximity/ light sensor changes?
# find elemnts, find next elements, check if session is in between?? find last element?

# A posteriori correction of session length when there is a call in the session
#'
#' Let S be a session. If there exists a phone call event in S, it is possible that the screen goes off
#' even though the session isn't over.
#' This function takes into consideration this possibility by not ending S if the call is not over.
